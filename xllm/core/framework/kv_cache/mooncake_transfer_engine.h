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

#pragma once

#include <Mooncake/mooncake-transfer-engine/include/transfer_engine.h>
#include <brpc/channel.h>
#include <brpc/server.h>

#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mooncake_transfer_engine.pb.h"
#include "platform/device.h"

namespace xllm {

using namespace mooncake;

class MooncakeTransferEngineService;

// Singleton core that holds the actual TransferEngine and brpc Server
// Multiple MooncakeTransferEngine instances share this core
class MooncakeTransferEngineCore {
 public:
  // Get the global singleton instance
  static MooncakeTransferEngineCore& get_instance() {
    static MooncakeTransferEngineCore instance;
    return instance;
  }

  // Initialize the core (only first call takes effect)
  bool initialize(int16_t listen_port, const torch::Device& device);

  // Get the underlying TransferEngine
  TransferEngine* engine() { return engine_.get(); }

  // Get the RPC address
  const std::string& addr() const { return addr_; }
  const std::string& host_ip() const { return host_ip_; }

  // Session management (shared across all MooncakeTransferEngine instances)
  bool open_session(const uint64_t cluster_id, const std::string& remote_addr);
  bool close_session(const uint64_t cluster_id, const std::string& remote_addr);
  SegmentHandle get_handle(const std::string& remote_addr);

  // RPC channel management
  proto::MooncakeTransferEngineService_Stub* get_or_create_stub(
      uint64_t cluster_id);

  void set_registered_memory_info(const std::vector<void*>& addrs,
                                  const std::vector<size_t>& lens,
                                  int64_t size_per_block,
                                  int64_t num_groups);
  void set_registered_memory_info(const std::vector<proto::BufferDesc>& buffers,
                                  int64_t num_groups);
  proto::RegisteredMemoryInfo get_registered_memory_info();

  bool is_initialized() const { return initialized_; }

 private:
  MooncakeTransferEngineCore() = default;
  ~MooncakeTransferEngineCore();
  MooncakeTransferEngineCore(const MooncakeTransferEngineCore&) = delete;
  MooncakeTransferEngineCore& operator=(const MooncakeTransferEngineCore&) =
      delete;

  std::mutex mutex_;
  bool initialized_ = false;

  std::string addr_;
  std::string host_ip_;
  int32_t rpc_port_ = 0;
  int16_t listen_port_ = 0;

  std::unique_ptr<TransferEngine> engine_;
  brpc::Server server_;
  std::shared_ptr<MooncakeTransferEngineService> service_;

  // Session handle with reference count for isolation between kv cache and
  // weight transfer
  struct SessionInfo {
    SegmentHandle handle;
    int ref_count = 0;
  };
  std::unordered_map<std::string, SessionInfo> handles_;
  std::unordered_map<uint64_t, proto::MooncakeTransferEngineService_Stub*>
      stub_map_;
};

class MooncakeTransferEngine final {
 public:
  enum class MoveOpcode { READ = 0, WRITE = 1 };

  struct BufferDesc {
    int64_t group_id = -1;
    int32_t slot_id = -1;
    uint64_t addr = 0;
    uint64_t len = 0;
    int64_t bytes_per_block = 0;
  };

  struct GroupDesc {
    std::map<int32_t, BufferDesc> slots;
  };

  struct MemInfo {
    std::vector<BufferDesc> buffers;
    std::vector<GroupDesc> groups;
    int64_t num_groups = 0;
  };

  MooncakeTransferEngine(const int16_t listen_port,
                         const torch::Device& device);
  virtual ~MooncakeTransferEngine() = default;

  std::string initialize();

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       int64_t size_per_block,
                       int64_t num_groups = -1);
  bool register_memory(const std::vector<BufferDesc>& buffers,
                       int64_t num_groups);

  bool move_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& group_ids,
                          MoveOpcode move_opcode);

  bool pull_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& group_ids);

  bool push_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& group_ids);

  // === XTensor mode: transfer by GlobalXTensor offsets ===
  // Instead of using block_id and per-layer buffers, this method uses
  // raw offsets into the GlobalXTensor memory region (buffer[0]).
  // src_offsets and dst_offsets are absolute offsets within GlobalXTensor.
  bool move_memory_by_global_offsets(const std::string& remote_addr,
                                     const std::vector<uint64_t>& src_offsets,
                                     const std::vector<uint64_t>& dst_offsets,
                                     size_t transfer_size,
                                     MoveOpcode move_opcode);

  bool open_session(const uint64_t cluster_id, const std::string& remote_addr);

  bool close_session(const uint64_t cluster_id, const std::string& remote_addr);

  proto::MooncakeTransferEngineService_Stub* create_rpc_channel(
      uint64_t cluster_id);

  bool fetch_remote_registered_memory(uint64_t cluster_id,
                                      const std::string& remote_addr);
  bool sync_local_memory_info_from_core();

  static bool parse_mem_info(const proto::RegisteredMemoryInfo& pb_info,
                             MemInfo* mem_info,
                             std::string* err);
  static bool same_layout(const MemInfo& local_info,
                          const MemInfo& remote_info,
                          std::string* err);
  static bool build_entries(const MemInfo& local_info,
                            const MemInfo& remote_info,
                            const std::vector<uint64_t>& src_blocks,
                            const std::vector<uint64_t>& dst_blocks,
                            const std::vector<uint64_t>& block_lens,
                            const std::vector<int64_t>& group_ids,
                            MoveOpcode move_opcode,
                            uint64_t remote_handle,
                            std::vector<TransferRequest>* entries,
                            std::string* err);

 private:
  int16_t listen_port_;
  // Only used by the legacy addrs/lens path.
  int64_t size_per_block_ = 0;
  int64_t num_groups_ = 0;
  Device device_;
  MooncakeTransferEngineCore& core_;
  MemInfo local_memory_info_;
  std::unordered_map<std::string, MemInfo> remote_memory_info_;
};

class MooncakeTransferEngineService
    : public proto::MooncakeTransferEngineService {
 public:
  MooncakeTransferEngineService() = default;

  virtual ~MooncakeTransferEngineService() = default;

  void set_registered_memory_info(const std::vector<void*>& addrs,
                                  const std::vector<size_t>& lens,
                                  int64_t size_per_block,
                                  int64_t num_groups);
  void set_registered_memory_info(const std::vector<proto::BufferDesc>& buffers,
                                  int64_t num_groups);
  proto::RegisteredMemoryInfo get_registered_memory_info();

  virtual void OpenSession(google::protobuf::RpcController* controller,
                           const proto::SessionInfo* request,
                           proto::Status* response,
                           google::protobuf::Closure* done) override;

  virtual void CloseSession(google::protobuf::RpcController* controller,
                            const proto::SessionInfo* request,
                            proto::Status* response,
                            google::protobuf::Closure* done) override;

  virtual void GetRegisteredMemory(google::protobuf::RpcController* controller,
                                   const proto::Empty* request,
                                   proto::RegisteredMemoryInfo* response,
                                   google::protobuf::Closure* done) override;

 private:
  std::mutex registered_memory_mutex_;
  proto::RegisteredMemoryInfo registered_memory_info_;
};

}  // namespace xllm
