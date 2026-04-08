/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mooncake_transfer_engine.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#endif
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>

#include "common/global_flags.h"
#include "util/net.h"

namespace xllm {

namespace {

#if defined(USE_NPU)
std::string get_transfer_engine_device_name(int32_t device_id) {
  int32_t phy_id = FLAGS_npu_phy_id;
  if (phy_id != -1) {
    device_id = phy_id;
  }
  return Device::type_str() + "_" + std::to_string(device_id);
}
#endif

enum class BufferLayoutKind {
  kUnknown = 0,
  kSinglePerGroup = 1,
  kPairPerGroup = 2
};

BufferLayoutKind detect_buffer_layout(size_t num_buffers, int64_t num_groups) {
  if (num_groups <= 0) {
    return BufferLayoutKind::kUnknown;
  }
  if (num_buffers == static_cast<size_t>(num_groups)) {
    return BufferLayoutKind::kSinglePerGroup;
  }
  if (num_buffers == static_cast<size_t>(num_groups * 2)) {
    return BufferLayoutKind::kPairPerGroup;
  }
  return BufferLayoutKind::kUnknown;
}

const char* buffer_layout_to_string(BufferLayoutKind layout) {
  switch (layout) {
    case BufferLayoutKind::kSinglePerGroup:
      return "single_per_group";
    case BufferLayoutKind::kPairPerGroup:
      return "pair_per_group";
    case BufferLayoutKind::kUnknown:
    default:
      return "unknown";
  }
}

std::string slot_id_to_string(int32_t slot_id) {
  return "slot=" + std::to_string(slot_id);
}

std::string format_buffer_desc(
    const MooncakeTransferEngine::BufferDesc& buffer_desc) {
  std::ostringstream os;
  os << "group=" << buffer_desc.group_id << ", "
     << slot_id_to_string(buffer_desc.slot_id) << ", addr=0x" << std::hex
     << buffer_desc.addr << std::dec << ", len=" << buffer_desc.len
     << ", bytes_per_block=" << buffer_desc.bytes_per_block;
  return os.str();
}

std::vector<proto::BufferDesc> to_proto_buffer_descs(
    const std::vector<MooncakeTransferEngine::BufferDesc>& buffers) {
  std::vector<proto::BufferDesc> proto_buffers;
  proto_buffers.reserve(buffers.size());
  for (const auto& buffer_desc : buffers) {
    proto::BufferDesc pb_desc;
    pb_desc.set_group_id(buffer_desc.group_id);
    pb_desc.set_slot_id(buffer_desc.slot_id);
    pb_desc.set_addr(buffer_desc.addr);
    pb_desc.set_len(buffer_desc.len);
    pb_desc.set_bytes_per_block(buffer_desc.bytes_per_block);
    proto_buffers.emplace_back(std::move(pb_desc));
  }
  return proto_buffers;
}

bool checked_multiply_u64(uint64_t lhs, uint64_t rhs, uint64_t* out) {
  if (out == nullptr) {
    return false;
  }
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool validate_buffer_range(const MooncakeTransferEngine::BufferDesc& buf,
                           uint64_t bias,
                           uint64_t len,
                           int64_t group_id,
                           int32_t slot_id,
                           uint64_t block_id,
                           uint64_t block_len,
                           const char* side,
                           std::string* err) {
  if (bias > buf.len || len > buf.len - bias) {
    if (err != nullptr) {
      std::ostringstream os;
      os << side << " range overflow, group=" << group_id << ", "
         << slot_id_to_string(slot_id) << ", block=" << block_id
         << ", block_len=" << block_len << ", bias=" << bias << ", len=" << len
         << ", buf_len=" << buf.len;
      *err = os.str();
    }
    return false;
  }
  return true;
}

uint64_t get_num_blocks(const MooncakeTransferEngine::BufferDesc& buf) {
  return static_cast<uint64_t>(
      buf.bytes_per_block > 0 ? buf.len / buf.bytes_per_block : 0);
}

}  // namespace

// ============================================================================
// MooncakeTransferEngineCore (Singleton)
// ============================================================================

MooncakeTransferEngineCore::~MooncakeTransferEngineCore() {
  // free stub
  for (auto& pair : stub_map_) {
    if (pair.second) {
      delete pair.second->channel();
      delete pair.second;
    }
  }
  stub_map_.clear();

  if (initialized_) {
    server_.Stop(0);
    server_.Join();
  }
}

bool MooncakeTransferEngineCore::initialize(int16_t listen_port,
                                            const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    LOG(INFO) << "MooncakeTransferEngineCore already initialized, reusing";
    return true;
  }

  listen_port_ = listen_port;
  host_ip_ = net::get_local_ip_addr();

  // Create TransferEngine
  engine_ = std::make_unique<TransferEngine>(true);

  Device dev(device);
  dev.set_device();
  dev.init_device_context();

  std::string hostname = host_ip_ + ":" + std::to_string(listen_port_);
#if defined(USE_NPU)
  int32_t device_id = dev.index();
  hostname += ":" + get_transfer_engine_device_name(device_id);
#endif

  if (engine_->init("P2PHANDSHAKE", hostname, "", 0)) {
    LOG(ERROR) << "engine init failed, hostname=" << hostname;
    return false;
  }

  LOG(INFO) << "TransferEngine init success, hostname=" << hostname;

  // Create brpc service and server
  service_ = std::make_shared<MooncakeTransferEngineService>();
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return false;
  }

  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server on port " << listen_port_;
    return false;
  }

  rpc_port_ = engine_->getRpcPort();
  addr_ = host_ip_ + ":" + std::to_string(rpc_port_);

  initialized_ = true;
  LOG(INFO) << "MooncakeTransferEngineCore initialize success, addr_=" << addr_;

  return true;
}

bool MooncakeTransferEngineCore::open_session(const uint64_t cluster_id,
                                              const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG(INFO) << "open_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (it != handles_.end()) {
    // Session exists, just increment ref count
    it->second.ref_count++;
    LOG(INFO) << "Reusing existing session for " << remote_addr
              << ", ref_count=" << it->second.ref_count;
    return true;
  }

  Transport::SegmentHandle handle;
  handle = engine_->openSegment(remote_addr);
  if (handle == (Transport::SegmentHandle)-1) {
    LOG(ERROR) << "Fail to connect to " << remote_addr;
    return false;
  }

  SessionInfo session_info;
  session_info.handle = handle;
  session_info.ref_count = 1;
  handles_[remote_addr] = session_info;

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService_Stub* stub =
        get_or_create_stub(cluster_id);
    if (!stub) {
      LOG(ERROR) << "create_rpc_channel failed";
      engine_->closeSegment(handle);
      handles_.erase(remote_addr);
      return false;
    }

    proto::SessionInfo proto_session_info;
    proto_session_info.set_addr(addr_);
    proto::Status status;
    brpc::Controller cntl;
    stub->OpenSession(&cntl, &proto_session_info, &status, nullptr);
    if (cntl.Failed() || !status.ok()) {
      LOG(ERROR) << "OpenSession failed, " << cntl.ErrorText();
      engine_->closeSegment(handle);
      handles_.erase(remote_addr);
      return false;
    }

    LOG(INFO) << "OpenSession RPC to " << remote_addr
              << ", local_addr=" << addr_
              << ", created local session handle=" << handle;
    return true;
  }

  LOG(INFO) << "Created new session for " << remote_addr << ", ref_count=1";

  return true;
}

bool MooncakeTransferEngineCore::close_session(const uint64_t cluster_id,
                                               const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG(INFO) << "close_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return true;
  }

  // Decrement ref count
  it->second.ref_count--;
  LOG(INFO) << "Decremented ref_count for " << remote_addr
            << ", ref_count=" << it->second.ref_count;

  // Only close when ref_count reaches 0
  if (it->second.ref_count > 0) {
    return true;
  }

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService_Stub* stub =
        get_or_create_stub(cluster_id);
    if (!stub) {
      LOG(ERROR) << "create_rpc_channel failed";
      return false;
    }

    proto::SessionInfo proto_session_info;
    proto_session_info.set_addr(addr_);

    proto::Status status;
    brpc::Controller cntl;
    stub->CloseSession(&cntl, &proto_session_info, &status, nullptr);
    if (cntl.Failed() || !status.ok()) {
      LOG(ERROR) << "CloseSession failed, " << cntl.ErrorText();
      return false;
    }
  }

  engine_->closeSegment(it->second.handle);
  handles_.erase(remote_addr);

  LOG(INFO) << "Closed session for " << remote_addr;

  return true;
}

SegmentHandle MooncakeTransferEngineCore::get_handle(
    const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return (SegmentHandle)-1;
  }
  return it->second.handle;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub(uint64_t cluster_id) {
  // Note: caller should hold mutex_ if needed
  auto it = stub_map_.find(cluster_id);
  if (it == stub_map_.end()) {
    auto [remote_ip, remote_port] = net::convert_uint64_to_ip_port(cluster_id);
    std::string remote_addr = remote_ip + ":" + std::to_string(remote_port);

    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = -1;
    std::string load_balancer = "";
    if (channel->Init(remote_addr.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << remote_addr;
      delete channel;
      return nullptr;
    }

    proto::MooncakeTransferEngineService_Stub* stub =
        new proto::MooncakeTransferEngineService_Stub(channel);
    stub_map_[cluster_id] = stub;
    return stub;
  }

  return it->second;
}

void MooncakeTransferEngineCore::set_registered_memory_info(
    const std::vector<void*>& addrs,
    const std::vector<size_t>& lens,
    int64_t size_per_block,
    int64_t num_groups) {
  CHECK(service_ != nullptr) << "MooncakeTransferEngineService not initialized";
  service_->set_registered_memory_info(addrs, lens, size_per_block, num_groups);
}

void MooncakeTransferEngineCore::set_registered_memory_info(
    const std::vector<proto::BufferDesc>& buffers,
    int64_t num_groups) {
  CHECK(service_ != nullptr) << "MooncakeTransferEngineService not initialized";
  service_->set_registered_memory_info(buffers, num_groups);
}

proto::RegisteredMemoryInfo
MooncakeTransferEngineCore::get_registered_memory_info() {
  CHECK(service_ != nullptr) << "MooncakeTransferEngineService not initialized";
  return service_->get_registered_memory_info();
}

// ============================================================================
// MooncakeTransferEngine
// ============================================================================

MooncakeTransferEngine::MooncakeTransferEngine(const int16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port),
      device_(device),
      core_(MooncakeTransferEngineCore::get_instance()) {}

std::string MooncakeTransferEngine::initialize() {
  if (!core_.initialize(listen_port_, device_)) {
    LOG(ERROR) << "Failed to initialize MooncakeTransferEngineCore";
    return "";
  }
  return core_.addr();
}

bool MooncakeTransferEngine::register_memory(std::vector<void*> addrs,
                                             std::vector<size_t> lens,
                                             int64_t size_per_block,
                                             int64_t num_groups) {
  int64_t num = addrs.size();
  const int64_t reg_groups =
      num_groups >= 0 ? num_groups : std::max<int64_t>(1, num / 2);

  size_per_block_ = size_per_block;
  proto::RegisteredMemoryInfo pb_info;
  for (void* addr : addrs) {
    pb_info.add_addrs(reinterpret_cast<uint64_t>(addr));
  }
  for (size_t len : lens) {
    pb_info.add_lens(static_cast<uint64_t>(len));
  }
  pb_info.set_size_per_block(size_per_block_);
  pb_info.set_num_layers(reg_groups);
  std::string err;
  MemInfo mem_info;
  if (!parse_mem_info(pb_info, &mem_info, &err)) {
    LOG(ERROR) << "register_memory parse_mem_info failed: " << err;
    return false;
  }

  std::vector<BufferEntry> buffers;
  buffers.reserve(num);
  for (size_t i = 0; i < num; i++) {
    buffers.emplace_back(addrs[i], lens[i]);
  }

  int ret =
      core_.engine()->registerLocalMemoryBatch(buffers, kWildcardLocation);
  if (ret) {
    LOG(ERROR) << "registerLocalMemoryBatch failed, ret=" << ret;
    return false;
  }

  num_groups_ = reg_groups;
  local_memory_info_ = std::move(mem_info);
  core_.set_registered_memory_info(addrs, lens, size_per_block_, num_groups_);

  const BufferLayoutKind layout =
      detect_buffer_layout(addrs.size(), num_groups_);

  LOG(INFO) << "register_memory success, size_per_block_=" << size_per_block_
            << ", num_groups=" << num_groups_ << ", num_buffers=" << num
            << ", layout=" << buffer_layout_to_string(layout);
  return true;
}

bool MooncakeTransferEngine::register_memory(
    const std::vector<BufferDesc>& buffers,
    int64_t num_groups) {
  const auto proto_buffers = to_proto_buffer_descs(buffers);
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(num_groups);
  for (const auto& pb_desc : proto_buffers) {
    *pb_info.add_buffers() = pb_desc;
  }

  std::string err;
  MemInfo mem_info;
  if (!parse_mem_info(pb_info, &mem_info, &err)) {
    LOG(ERROR) << "register_memory parse_mem_info failed: " << err;
    return false;
  }

  std::vector<BufferEntry> entries;
  entries.reserve(buffers.size());
  for (const auto& buffer_desc : mem_info.buffers) {
    entries.emplace_back(reinterpret_cast<void*>(buffer_desc.addr),
                         static_cast<size_t>(buffer_desc.len));
  }

  int ret =
      core_.engine()->registerLocalMemoryBatch(entries, kWildcardLocation);
  if (ret) {
    LOG(ERROR) << "registerLocalMemoryBatch failed, ret=" << ret;
    return false;
  }

  num_groups_ = num_groups;
  local_memory_info_ = std::move(mem_info);
  core_.set_registered_memory_info(proto_buffers, num_groups_);

  LOG(INFO) << "register_memory success, num_groups=" << num_groups_
            << ", num_buffers=" << buffers.size() << ", layout=per_desc";
  return true;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngine::create_rpc_channel(uint64_t cluster_id) {
  return core_.get_or_create_stub(cluster_id);
}

bool MooncakeTransferEngine::fetch_remote_registered_memory(
    uint64_t cluster_id,
    const std::string& remote_addr) {
  auto* stub = create_rpc_channel(cluster_id);
  if (!stub) {
    LOG(ERROR) << "Failed to create rpc channel for registered memory"
               << ", cluster_id=" << cluster_id;
    return false;
  }

  proto::Empty request;
  proto::RegisteredMemoryInfo response;
  brpc::Controller cntl;
  stub->GetRegisteredMemory(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetRegisteredMemory failed, remote_addr=" << remote_addr
               << ", cluster_id=" << cluster_id
               << ", error=" << cntl.ErrorText();
    return false;
  }
  if (response.addrs_size() != response.lens_size()) {
    LOG(ERROR) << "GetRegisteredMemory returned mismatched addrs/lens"
               << ", remote_addr=" << remote_addr
               << ", addrs_size=" << response.addrs_size()
               << ", lens_size=" << response.lens_size();
    return false;
  }

  MemInfo info;
  std::string err;
  if (!parse_mem_info(response, &info, &err)) {
    LOG(ERROR) << "GetRegisteredMemory parse failed, remote_addr="
               << remote_addr << ", err=" << err;
    return false;
  }
  if (!same_layout(local_memory_info_, info, &err)) {
    LOG(ERROR) << "GetRegisteredMemory mem info mismatch, remote_addr="
               << remote_addr << ", err=" << err;
    return false;
  }
  for (int64_t group_id = 0; group_id < local_memory_info_.num_groups;
       ++group_id) {
    const auto& local_slots = local_memory_info_.groups[group_id].slots;
    const auto& remote_slots = info.groups[group_id].slots;
    for (const auto& [slot_id, local_buf] : local_slots) {
      const auto remote_it = remote_slots.find(slot_id);
      if (remote_it == remote_slots.end()) {
        continue;
      }
      const auto local_blocks = get_num_blocks(local_buf);
      const auto remote_blocks = get_num_blocks(remote_it->second);
      if (local_blocks == remote_blocks) {
        continue;
      }
      LOG(INFO) << "GetRegisteredMemory capacity asymmetry accepted"
                << ", remote_addr=" << remote_addr << ", group=" << group_id
                << ", " << slot_id_to_string(slot_id)
                << ", local_blocks=" << local_blocks
                << ", remote_blocks=" << remote_blocks
                << ", local_bytes_per_block=" << local_buf.bytes_per_block
                << ", remote_bytes_per_block="
                << remote_it->second.bytes_per_block;
    }
  }

  remote_memory_info_[remote_addr] = std::move(info);
  return true;
}

bool MooncakeTransferEngine::sync_local_memory_info_from_core() {
  const auto pb_info = core_.get_registered_memory_info();
  std::string err;
  MemInfo mem_info;
  if (!parse_mem_info(pb_info, &mem_info, &err)) {
    LOG(ERROR) << "sync_local_memory_info_from_core parse_mem_info failed: "
               << err;
    return false;
  }

  num_groups_ = mem_info.num_groups;
  local_memory_info_ = std::move(mem_info);
  return true;
}

bool MooncakeTransferEngine::open_session(const uint64_t cluster_id,
                                          const std::string& remote_addr) {
  if (!core_.open_session(cluster_id, remote_addr)) {
    return false;
  }
  if (!fetch_remote_registered_memory(cluster_id, remote_addr)) {
    LOG(ERROR) << "Failed to fetch remote registered memory, remote_addr="
               << remote_addr << ", cluster_id=" << cluster_id;
    core_.close_session(cluster_id, remote_addr);
    return false;
  }
  return true;
}

bool MooncakeTransferEngine::close_session(const uint64_t cluster_id,
                                           const std::string& remote_addr) {
  remote_memory_info_.erase(remote_addr);
  return core_.close_session(cluster_id, remote_addr);
}

// Merge the source and destination block ids into a single block when both are
// consecutive.
void merge_block_ids(const std::vector<uint64_t>& src_blocks,
                     const std::vector<uint64_t>& dst_blocks,
                     std::vector<uint64_t>& merged_src_blocks,
                     std::vector<uint64_t>& merged_dst_blocks,
                     std::vector<uint64_t>& block_lengths) {
  // Create an index array and sort it based on the values of src blocks.
  size_t block_num = src_blocks.size();
  if (block_num == 0) {
    return;
  }
  std::vector<uint64_t> indices(block_num);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(), indices.end(), [&src_blocks](uint64_t i, uint64_t j) {
        return src_blocks[i] < src_blocks[j];
      });

  // Generate sorted src blocks and dst blocks.
  std::vector<uint64_t> sorted_src_blocks;
  std::vector<uint64_t> sorted_dst_blocks;
  sorted_src_blocks.reserve(block_num);
  sorted_dst_blocks.reserve(block_num);
  for (auto id : indices) {
    sorted_src_blocks.emplace_back(src_blocks[id]);
    sorted_dst_blocks.emplace_back(dst_blocks[id]);
  }

  // Obtain continuous blocks.
  uint64_t current_src_id = sorted_src_blocks[0];
  uint64_t current_dst_id = sorted_dst_blocks[0];
  uint64_t current_length = 1;
  merged_src_blocks.reserve(block_num);
  merged_dst_blocks.reserve(block_num);
  block_lengths.reserve(block_num);
  for (size_t i = 1; i < sorted_src_blocks.size(); ++i) {
    if (sorted_src_blocks[i] == sorted_src_blocks[i - 1] + 1 &&
        sorted_dst_blocks[i] == sorted_dst_blocks[i - 1] + 1) {
      current_length++;
    } else {
      merged_src_blocks.emplace_back(current_src_id);
      merged_dst_blocks.emplace_back(current_dst_id);
      block_lengths.emplace_back(current_length);
      current_src_id = sorted_src_blocks[i];
      current_dst_id = sorted_dst_blocks[i];
      current_length = 1;
    }
  }
  merged_src_blocks.emplace_back(current_src_id);
  merged_dst_blocks.emplace_back(current_dst_id);
  block_lengths.emplace_back(current_length);
}

bool MooncakeTransferEngine::move_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& group_ids,
    MoveOpcode move_opcode) {
  if (src_blocks.size() != dst_blocks.size()) {
    LOG(ERROR) << "move_memory_blocks block count mismatch, src_blocks.size="
               << src_blocks.size() << ", dst_blocks.size=" << dst_blocks.size()
               << ", move_opcode=" << static_cast<int>(move_opcode);
    return false;
  }

  auto remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == (SegmentHandle)-1) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return false;
  }

  auto* engine = core_.engine();
  const auto remote_it = remote_memory_info_.find(remote_addr);
  if (remote_it == remote_memory_info_.end()) {
    LOG(ERROR) << "remote registered memory does not exist: " << remote_addr;
    return false;
  }
  const MemInfo& remote_memory_info = remote_it->second;

  // Merge consecutive block ids to improve transmission efficiency.
  std::vector<uint64_t> merged_src_blocks;
  std::vector<uint64_t> merged_dst_blocks;
  std::vector<uint64_t> block_lengths;
  merge_block_ids(src_blocks,
                  dst_blocks,
                  merged_src_blocks,
                  merged_dst_blocks,
                  block_lengths);

  std::vector<int64_t> target_group_ids;
  if (group_ids.empty()) {
    target_group_ids.resize(num_groups_);
    std::iota(target_group_ids.begin(), target_group_ids.end(), 0);
  } else {
    target_group_ids = group_ids;
  }

  std::vector<TransferRequest> entries;
  std::string err;
  if (!build_entries(local_memory_info_,
                     remote_memory_info,
                     merged_src_blocks,
                     merged_dst_blocks,
                     block_lengths,
                     target_group_ids,
                     move_opcode,
                     remote_handle,
                     &entries,
                     &err)) {
    LOG(ERROR) << "move_memory_blocks build_entries failed, remote_addr="
               << remote_addr << ", err=" << err;
    return false;
  }

  auto batch_size = entries.size();
  auto batch_id = engine->allocateBatchID(batch_size);
  mooncake::Status s = engine->submitTransfer(batch_id, entries);
  if (!s.ok()) {
    LOG(ERROR) << "submit failed";
    engine->freeBatchID(batch_id);
    return false;
  }

  TransferStatus status;
  bool completed = false;
  while (!completed) {
    s = engine->getBatchTransferStatus(batch_id, status);
    if (!s.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      completed = true;
    }

    if (status.s == TransferStatusEnum::COMPLETED) {
      completed = true;
    } else if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      completed = true;
    } else if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      completed = true;
    }
  }

  s = engine->freeBatchID(batch_id);
  if (!s.ok()) {
    LOG(ERROR) << "freeBatchID failed";
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::parse_mem_info(
    const proto::RegisteredMemoryInfo& pb_info,
    MemInfo* mem_info,
    std::string* err) {
  if (mem_info == nullptr) {
    if (err != nullptr) {
      *err = "mem_info is null";
    }
    return false;
  }

  mem_info->buffers.clear();
  mem_info->groups.clear();
  mem_info->num_groups = pb_info.num_layers();

  if (mem_info->num_groups <= 0) {
    if (err != nullptr) {
      *err = "num_groups must be positive";
    }
    return false;
  }

  mem_info->groups.assign(mem_info->num_groups, GroupDesc{});
  auto add_desc = [mem_info, err](const BufferDesc& buffer_desc) {
    if (buffer_desc.group_id < 0 ||
        buffer_desc.group_id >= mem_info->num_groups) {
      if (err != nullptr) {
        *err =
            "buffer group_id out of range: " + format_buffer_desc(buffer_desc);
      }
      return false;
    }
    if (buffer_desc.slot_id < 0) {
      if (err != nullptr) {
        *err = "buffer slot_id must be non-negative: " +
               format_buffer_desc(buffer_desc);
      }
      return false;
    }
    if (buffer_desc.bytes_per_block <= 0) {
      if (err != nullptr) {
        *err = "buffer bytes_per_block must be positive: " +
               format_buffer_desc(buffer_desc);
      }
      return false;
    }
    if (buffer_desc.len == 0) {
      if (err != nullptr) {
        *err =
            "buffer len must be positive: " + format_buffer_desc(buffer_desc);
      }
      return false;
    }
    if (buffer_desc.len % buffer_desc.bytes_per_block != 0) {
      if (err != nullptr) {
        *err = "buffer len must align with bytes_per_block: " +
               format_buffer_desc(buffer_desc);
      }
      return false;
    }
    auto& slots = mem_info->groups[buffer_desc.group_id].slots;
    if (slots.find(buffer_desc.slot_id) != slots.end()) {
      if (err != nullptr) {
        *err = "duplicate buffer desc: " + format_buffer_desc(buffer_desc);
      }
      return false;
    }
    slots.emplace(buffer_desc.slot_id, buffer_desc);
    mem_info->buffers.push_back(buffer_desc);
    return true;
  };

  if (pb_info.buffers_size() > 0) {
    for (const auto& pb_desc : pb_info.buffers()) {
      BufferDesc buffer_desc;
      buffer_desc.group_id = pb_desc.group_id();
      buffer_desc.slot_id = pb_desc.slot_id();
      buffer_desc.addr = pb_desc.addr();
      buffer_desc.len = pb_desc.len();
      buffer_desc.bytes_per_block = pb_desc.bytes_per_block();
      if (!add_desc(buffer_desc)) {
        return false;
      }
    }
  } else {
    if (pb_info.addrs_size() != pb_info.lens_size()) {
      if (err != nullptr) {
        *err = "legacy addrs/lens size mismatch";
      }
      return false;
    }
    const BufferLayoutKind layout =
        detect_buffer_layout(pb_info.addrs_size(), mem_info->num_groups);
    if (layout == BufferLayoutKind::kUnknown) {
      if (err != nullptr) {
        *err = "legacy layout unknown";
      }
      return false;
    }
    if (pb_info.size_per_block() <= 0) {
      if (err != nullptr) {
        *err = "legacy size_per_block must be positive";
      }
      return false;
    }

    for (int64_t group_id = 0; group_id < mem_info->num_groups; ++group_id) {
      BufferDesc buffer_desc;
      buffer_desc.group_id = group_id;
      buffer_desc.slot_id = 0;
      buffer_desc.addr = pb_info.addrs(group_id);
      buffer_desc.len = pb_info.lens(group_id);
      buffer_desc.bytes_per_block = pb_info.size_per_block();
      if (!add_desc(buffer_desc)) {
        return false;
      }
    }
    if (layout == BufferLayoutKind::kPairPerGroup) {
      for (int64_t group_id = 0; group_id < mem_info->num_groups; ++group_id) {
        const int64_t idx = group_id + mem_info->num_groups;
        BufferDesc buffer_desc;
        buffer_desc.group_id = group_id;
        buffer_desc.slot_id = 1;
        buffer_desc.addr = pb_info.addrs(idx);
        buffer_desc.len = pb_info.lens(idx);
        buffer_desc.bytes_per_block = pb_info.size_per_block();
        if (!add_desc(buffer_desc)) {
          return false;
        }
      }
    }
  }

  std::sort(mem_info->buffers.begin(),
            mem_info->buffers.end(),
            [](const BufferDesc& lhs, const BufferDesc& rhs) {
              return std::tie(lhs.group_id, lhs.slot_id) <
                     std::tie(rhs.group_id, rhs.slot_id);
            });
  return true;
}

bool MooncakeTransferEngine::same_layout(const MemInfo& local_info,
                                         const MemInfo& remote_info,
                                         std::string* err) {
  if (local_info.num_groups != remote_info.num_groups) {
    if (err != nullptr) {
      *err = "num_groups mismatch, local=" +
             std::to_string(local_info.num_groups) +
             ", remote=" + std::to_string(remote_info.num_groups);
    }
    return false;
  }

  for (int64_t group_id = 0; group_id < local_info.num_groups; ++group_id) {
    const auto& local_slots = local_info.groups[group_id].slots;
    const auto& remote_slots = remote_info.groups[group_id].slots;
    if (local_slots.size() != remote_slots.size()) {
      if (err != nullptr) {
        *err = "slot count mismatch at group=" + std::to_string(group_id) +
               ", local=" + std::to_string(local_slots.size()) +
               ", remote=" + std::to_string(remote_slots.size());
      }
      return false;
    }
    for (const auto& [slot_id, local_buf] : local_slots) {
      const auto remote_it = remote_slots.find(slot_id);
      if (remote_it == remote_slots.end()) {
        if (err != nullptr) {
          *err = "slot mismatch at group=" + std::to_string(group_id) + ", " +
                 slot_id_to_string(slot_id);
        }
        return false;
      }
      if (local_buf.bytes_per_block != remote_it->second.bytes_per_block) {
        if (err != nullptr) {
          *err =
              "bytes_per_block mismatch at group=" + std::to_string(group_id) +
              ", " + slot_id_to_string(slot_id) +
              ", local=" + std::to_string(local_buf.bytes_per_block) +
              ", remote=" + std::to_string(remote_it->second.bytes_per_block);
        }
        return false;
      }
    }
  }
  return true;
}

bool MooncakeTransferEngine::build_entries(
    const MemInfo& local_info,
    const MemInfo& remote_info,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& block_lens,
    const std::vector<int64_t>& group_ids,
    MoveOpcode move_opcode,
    uint64_t remote_handle,
    std::vector<TransferRequest>* entries,
    std::string* err) {
  if (entries == nullptr) {
    if (err != nullptr) {
      *err = "entries is null";
    }
    return false;
  }
  if (src_blocks.size() != dst_blocks.size() ||
      src_blocks.size() != block_lens.size()) {
    if (err != nullptr) {
      *err = "merged block arrays size mismatch";
    }
    return false;
  }

  if (!same_layout(local_info, remote_info, err)) {
    return false;
  }
  TransferRequest::OpCode opcode = move_opcode == MoveOpcode::WRITE
                                       ? TransferRequest::WRITE
                                       : TransferRequest::READ;

  const std::vector<int64_t>* target_groups = &group_ids;
  std::vector<int64_t> all_groups;
  if (group_ids.empty()) {
    all_groups.resize(local_info.num_groups);
    std::iota(all_groups.begin(), all_groups.end(), 0);
    target_groups = &all_groups;
  }

  entries->clear();
  entries->reserve(local_info.buffers.size() * src_blocks.size());
  for (const auto group_id : *target_groups) {
    if (group_id < 0 || group_id >= local_info.num_groups) {
      if (err != nullptr) {
        *err = "group_id out of range: " + std::to_string(group_id);
      }
      return false;
    }
    const auto& local_slots = local_info.groups[group_id].slots;
    const auto& remote_slots = remote_info.groups[group_id].slots;
    for (const auto& [slot_id, local_buf] : local_slots) {
      const auto remote_it = remote_slots.find(slot_id);
      if (remote_it == remote_slots.end()) {
        if (err != nullptr) {
          *err = "missing remote buffer at group=" + std::to_string(group_id) +
                 ", " + slot_id_to_string(slot_id);
        }
        return false;
      }
      const auto& remote_buf = remote_it->second;
      const auto bytes_per_block =
          static_cast<uint64_t>(local_buf.bytes_per_block);
      auto* local_base = reinterpret_cast<char*>(local_buf.addr);
      auto* remote_base = reinterpret_cast<char*>(remote_buf.addr);
      for (size_t idx = 0; idx < src_blocks.size(); ++idx) {
        if (block_lens[idx] == 0) {
          if (err != nullptr) {
            *err = "block_len must be positive, group=" +
                   std::to_string(group_id) + ", " + slot_id_to_string(slot_id);
          }
          return false;
        }
        uint64_t src_bias = 0;
        uint64_t dst_bias = 0;
        uint64_t len = 0;
        if (!checked_multiply_u64(
                src_blocks[idx], bytes_per_block, &src_bias) ||
            !checked_multiply_u64(
                dst_blocks[idx], bytes_per_block, &dst_bias) ||
            !checked_multiply_u64(block_lens[idx], bytes_per_block, &len)) {
          if (err != nullptr) {
            *err = "entry size overflow, group=" + std::to_string(group_id) +
                   ", " + slot_id_to_string(slot_id) +
                   ", src_block=" + std::to_string(src_blocks[idx]) +
                   ", dst_block=" + std::to_string(dst_blocks[idx]) +
                   ", block_len=" + std::to_string(block_lens[idx]);
          }
          return false;
        }
        const uint64_t local_bias =
            move_opcode == MoveOpcode::READ ? dst_bias : src_bias;
        const uint64_t remote_bias =
            move_opcode == MoveOpcode::READ ? src_bias : dst_bias;
        if (!validate_buffer_range(local_buf,
                                   local_bias,
                                   len,
                                   group_id,
                                   slot_id,
                                   move_opcode == MoveOpcode::READ
                                       ? dst_blocks[idx]
                                       : src_blocks[idx],
                                   block_lens[idx],
                                   "local",
                                   err) ||
            !validate_buffer_range(remote_it->second,
                                   remote_bias,
                                   len,
                                   group_id,
                                   slot_id,
                                   move_opcode == MoveOpcode::READ
                                       ? src_blocks[idx]
                                       : dst_blocks[idx],
                                   block_lens[idx],
                                   "remote",
                                   err)) {
          return false;
        }
        TransferRequest& entry = entries->emplace_back();
        entry.opcode = opcode;
        entry.length = len;
        entry.source = static_cast<void*>(local_base + local_bias);
        entry.target_id = remote_handle;
        entry.target_offset =
            reinterpret_cast<uint64_t>(remote_base + remote_bias);
        entry.advise_retry_cnt = 0;
      }
    }
  }

  return true;
}

bool MooncakeTransferEngine::move_memory_by_global_offsets(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_offsets,
    const std::vector<uint64_t>& dst_offsets,
    size_t transfer_size,
    MoveOpcode move_opcode) {
  auto remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == (SegmentHandle)-1) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return false;
  }

  auto* engine = core_.engine();
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc;
  remote_segment_desc =
      engine->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return false;
  }

  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc;
  local_segment_desc =
      engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return false;
  }

  // XTensor mode: use buffer[0] which is the GlobalXTensor
  if (local_segment_desc->buffers.empty() ||
      remote_segment_desc->buffers.empty()) {
    LOG(ERROR) << "No buffers registered for XTensor mode";
    return false;
  }

  char* local_base =
      reinterpret_cast<char*>(local_segment_desc->buffers[0].addr);
  char* remote_base =
      reinterpret_cast<char*>(remote_segment_desc->buffers[0].addr);

  TransferRequest::OpCode opcode;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  } else {
    opcode = TransferRequest::READ;
  }

  std::vector<TransferRequest> entries;
  entries.reserve(src_offsets.size());

  for (size_t i = 0; i < src_offsets.size(); ++i) {
    const uint64_t local_offset =
        move_opcode == MoveOpcode::READ ? dst_offsets[i] : src_offsets[i];
    const uint64_t remote_offset =
        move_opcode == MoveOpcode::READ ? src_offsets[i] : dst_offsets[i];
    TransferRequest& entry = entries.emplace_back();
    entry.opcode = opcode;
    entry.length = transfer_size;
    entry.source = static_cast<void*>(local_base + local_offset);
    entry.target_id = remote_handle;
    entry.target_offset =
        reinterpret_cast<uint64_t>(remote_base + remote_offset);
    entry.advise_retry_cnt = 0;
  }

  auto batch_size = entries.size();
  auto batch_id = engine->allocateBatchID(batch_size);
  mooncake::Status s = engine->submitTransfer(batch_id, entries);
  if (!s.ok()) {
    LOG(ERROR) << "submit failed in move_memory_by_global_offsets";
    engine->freeBatchID(batch_id);
    return false;
  }

  TransferStatus status;
  bool completed = false;
  while (!completed) {
    s = engine->getBatchTransferStatus(batch_id, status);
    if (!s.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      completed = true;
    }

    if (status.s == TransferStatusEnum::COMPLETED) {
      completed = true;
    } else if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      completed = true;
    } else if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      completed = true;
    }
  }

  s = engine->freeBatchID(batch_id);
  if (!s.ok()) {
    LOG(ERROR) << "freeBatchID failed";
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::pull_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& group_ids) {
  auto ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, group_ids, MoveOpcode::READ);
  if (!ret) {
    LOG(ERROR) << "Pull memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::push_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& group_ids) {
  auto ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, group_ids, MoveOpcode::WRITE);
  if (!ret) {
    LOG(ERROR) << "Push memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

// ============================================================================
// MooncakeTransferEngineService
// ============================================================================

void MooncakeTransferEngineService::set_registered_memory_info(
    const std::vector<void*>& addrs,
    const std::vector<size_t>& lens,
    int64_t size_per_block,
    int64_t num_groups) {
  std::lock_guard<std::mutex> lock(registered_memory_mutex_);
  registered_memory_info_.clear_addrs();
  registered_memory_info_.clear_lens();
  for (const auto* addr : addrs) {
    registered_memory_info_.add_addrs(reinterpret_cast<uint64_t>(addr));
  }
  for (const size_t len : lens) {
    registered_memory_info_.add_lens(static_cast<uint64_t>(len));
  }
  registered_memory_info_.set_size_per_block(size_per_block);
  registered_memory_info_.set_num_layers(num_groups);
  registered_memory_info_.clear_buffers();
}

void MooncakeTransferEngineService::set_registered_memory_info(
    const std::vector<proto::BufferDesc>& buffers,
    int64_t num_groups) {
  std::lock_guard<std::mutex> lock(registered_memory_mutex_);
  registered_memory_info_.clear_addrs();
  registered_memory_info_.clear_lens();
  registered_memory_info_.set_size_per_block(0);
  registered_memory_info_.set_num_layers(num_groups);
  registered_memory_info_.clear_buffers();
  for (const auto& buffer_desc : buffers) {
    *registered_memory_info_.add_buffers() = buffer_desc;
  }
}

proto::RegisteredMemoryInfo
MooncakeTransferEngineService::get_registered_memory_info() {
  std::lock_guard<std::mutex> lock(registered_memory_mutex_);
  return registered_memory_info_;
}

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().open_session(0, remote_addr);

  response->set_ok(result);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().close_session(0, remote_addr);

  response->set_ok(result);
}

void MooncakeTransferEngineService::GetRegisteredMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::RegisteredMemoryInfo* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  *response = get_registered_memory_info();
}

}  // namespace xllm
