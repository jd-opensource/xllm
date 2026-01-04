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

#include <acl/acl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <numeric>

#include "common/global_flags.h"
#include "util/net.h"

namespace xllm {

MooncakeTransferEngine::MooncakeTransferEngine(const int16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port), device_(device) {
  engine_ = std::make_unique<TransferEngine>(true);

  host_ip_ = net::get_local_ip_addr();

  device_.set_device();
  device_.init_device_context();

  int32_t device_id = device_.index();
  std::string hostname;
  int32_t phy_id = FLAGS_npu_phy_id;
  if (phy_id != -1) {
    hostname = host_ip_ + ":" + std::to_string(listen_port_) + ":npu_" +
               std::to_string(phy_id);
  } else {
    hostname = host_ip_ + ":" + std::to_string(listen_port_) + ":npu_" +
               std::to_string(device_id);
  }

  if (engine_->init("P2PHANDSHAKE", hostname, "", 0)) {
    LOG(ERROR) << "engine init failed, hostname=" << hostname;
  }

  LOG(INFO) << "engine init success, hostname=" << hostname;
}

MooncakeTransferEngine::~MooncakeTransferEngine() {
  // free stub
  for (auto& pair : stub_map_) {
    if (pair.second) {
      delete pair.second->channel();
      delete pair.second;
    }
  }
  stub_map_.clear();

  if (has_initialized_) {
    server_.Stop(0);
    server_.Join();
  }
}

std::string MooncakeTransferEngine::initialize() {
  service_ = std::make_shared<MooncakeTransferEngineService>(this);
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return "";
  }

  // Start the server.
  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server";
    return "";
  }

  has_initialized_ = true;

  rpc_port_ = engine_->getRpcPort();
  addr_ = host_ip_ + ":" + std::to_string(rpc_port_);
  LOG(INFO) << "MooncakeTransferEngine initialize success, addr_=" << addr_;

  return addr_;
}

bool MooncakeTransferEngine::register_memory(std::vector<void*> addrs,
                                             std::vector<size_t> lens,
                                             int64_t size_per_block) {
  int64_t num = addrs.size();
  num_layers_ = num / 2;

  std::vector<BufferEntry> buffers;
  buffers.reserve(num);
  for (size_t i = 0; i < num; i++) {
    buffers.push_back(BufferEntry{(void*)addrs[i], lens[i]});
  }

  int ret = engine_->registerLocalMemoryBatch(buffers, kWildcardLocation);
  if (ret) {
    LOG(ERROR) << "registerLocalMemoryBatch failed, ret=" << ret;
    return false;
  }

  size_per_block_ = size_per_block;

  LOG(INFO) << "register_memory success, size_per_block_=" << size_per_block_;

  return true;
}

proto::MooncakeTransferEngineService*
MooncakeTransferEngine::create_rpc_channel(uint64_t cluster_id) {
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

bool MooncakeTransferEngine::open_session(const uint64_t cluster_id,
                                          const std::string& remote_addr) {
  LOG(INFO) << "open_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (it != handles_.end()) {
    return true;
  }

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService* stub = create_rpc_channel(cluster_id);
    if (!stub) {
      LOG(ERROR) << "create_rpc_channel failed";
      return false;
    }

    proto::SessionInfo proto_session_info;
    proto_session_info.set_addr(addr_);

    proto::Status status;
    brpc::Controller cntl;
    stub->OpenSession(&cntl, &proto_session_info, &status, nullptr);
    if (cntl.Failed() || !status.ok()) {
      LOG(ERROR) << "OpenSession failed, " << cntl.ErrorText();
      return false;
    }
  }

  Transport::SegmentHandle handle;
  handle = engine_->openSegment(remote_addr);
  if (handle == (Transport::SegmentHandle)-1) {
    LOG(ERROR) << "Fail to connect to " << remote_addr;
    return false;
  }

  handles_[remote_addr] = handle;

  return true;
}

bool MooncakeTransferEngine::close_session(const uint64_t cluster_id,
                                           const std::string& remote_addr) {
  LOG(INFO) << "close_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return true;
  }

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService* stub = create_rpc_channel(cluster_id);
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

  engine_->closeSegment(it->second);

  handles_.erase(remote_addr);

  return true;
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
    const std::vector<int64_t>& layer_ids,
    MoveOpcode move_opcode) {
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    LOG(ERROR) << "remote addr does not exist" << remote_addr;
    return false;
  }

  auto remote_handle = it->second;
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc;
  remote_segment_desc =
      engine_->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return false;
  }

  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc;
  local_segment_desc =
      engine_->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return false;
  }

  // Merge consecutive block ids to improve transmission efficiency.
  std::vector<uint64_t> merged_src_blocks;
  std::vector<uint64_t> merged_dst_blocks;
  std::vector<uint64_t> block_lengths;
  merge_block_ids(src_blocks,
                  dst_blocks,
                  merged_src_blocks,
                  merged_dst_blocks,
                  block_lengths);

  std::vector<int64_t> addr_ids;
  if (layer_ids.size() == 0) {
    addr_ids.resize(num_layers_);
    std::iota(addr_ids.begin(), addr_ids.end(), 0);
  } else {
    addr_ids = layer_ids;
  }

  TransferRequest::OpCode opcode;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  } else {
    opcode = TransferRequest::READ;
  }

  std::vector<TransferRequest> entries;
  for (auto addr_id : addr_ids) {
    char* k_local_base = (char*)(local_segment_desc->buffers[addr_id].addr);
    char* k_remote_base = (char*)(remote_segment_desc->buffers[addr_id].addr);

    int64_t v_addr_id = addr_id + num_layers_;
    char* v_local_base = (char*)(local_segment_desc->buffers[v_addr_id].addr);
    char* v_remote_base = (char*)(remote_segment_desc->buffers[v_addr_id].addr);

    for (size_t i = 0; i < merged_src_blocks.size(); ++i) {
      uint64_t src_block_id = merged_src_blocks[i];
      uint64_t dst_block_id = merged_dst_blocks[i];
      uint64_t block_length = block_lengths[i];
      uint64_t src_bias = src_block_id * size_per_block_;
      uint64_t dst_bias = dst_block_id * size_per_block_;
      uint64_t len = block_length * size_per_block_;

      TransferRequest k_entry;
      k_entry.opcode = opcode;
      k_entry.length = len;
      k_entry.source = (void*)(k_local_base + src_bias);
      k_entry.target_id = remote_handle;
      k_entry.target_offset = (uint64_t)(k_remote_base + dst_bias);
      k_entry.advise_retry_cnt = 0;
      entries.push_back(k_entry);

      TransferRequest v_entry;
      v_entry.opcode = opcode;
      v_entry.length = len;
      v_entry.source = (void*)(v_local_base + src_bias);
      v_entry.target_id = remote_handle;
      v_entry.target_offset = (uint64_t)(v_remote_base + dst_bias);
      v_entry.advise_retry_cnt = 0;
      entries.push_back(v_entry);
    }
  }

  auto batch_size = entries.size();
  auto batch_id = engine_->allocateBatchID(batch_size);
  mooncake::Status s = engine_->submitTransfer(batch_id, entries);
  if (!s.ok()) {
    LOG(ERROR) << "submit failed";
    engine_->freeBatchID(batch_id);
    return false;
  }

  TransferStatus status;
  bool completed = false;
  while (!completed) {
    s = engine_->getBatchTransferStatus(batch_id, status);
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

  s = engine_->freeBatchID(batch_id);
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
    const std::vector<int64_t>& layer_ids) {
  auto ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, layer_ids, MoveOpcode::READ);
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
    const std::vector<int64_t>& layer_ids) {
  auto ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, layer_ids, MoveOpcode::WRITE);
  if (!ret) {
    LOG(ERROR) << "Push memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

MooncakeTransferEngineService::MooncakeTransferEngineService(
    MooncakeTransferEngine* mooncake_te)
    : mooncake_te_(mooncake_te) {};

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  bool result = mooncake_te_->open_session(0, remote_addr);

  response->set_ok(result);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  bool result = mooncake_te_->close_session(0, remote_addr);

  response->set_ok(result);
}

}  // namespace xllm
