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

#include <thread>

#include "mooncake_transfer_engine.pb.h"
#include "platform/device.h"

namespace xllm {

using namespace mooncake;

class MooncakeTransferEngineService;

class MooncakeTransferEngine final {
 public:
  enum class MoveOpcode { READ = 0, WRITE = 1 };

  MooncakeTransferEngine(const int16_t listen_port,
                         const torch::Device& device);
  virtual ~MooncakeTransferEngine();

  std::string initialize();

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       int64_t size_per_block);

  bool move_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& layer_ids,
                          MoveOpcode move_opcode);

  bool pull_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& layer_ids);

  bool push_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& layer_ids);

  bool open_session(const uint64_t cluster_id, const std::string& remote_addr);

  bool close_session(const uint64_t cluster_id, const std::string& remote_addr);

  proto::MooncakeTransferEngineService* create_rpc_channel(uint64_t cluster_id);

 private:
  std::string addr_;
  std::string host_ip_;
  int32_t rpc_port_;
  int16_t listen_port_;
  int64_t size_per_block_;
  int64_t num_layers_;

  Device device_;

  std::unordered_map<std::string, SegmentHandle> handles_;

  brpc::Server server_;
  std::shared_ptr<MooncakeTransferEngineService> service_;
  bool has_initialized_ = false;
  std::unordered_map<uint64_t, proto::MooncakeTransferEngineService_Stub*>
      stub_map_;

  std::unique_ptr<TransferEngine> engine_;
};

class MooncakeTransferEngineService
    : public proto::MooncakeTransferEngineService {
 public:
  MooncakeTransferEngineService(MooncakeTransferEngine* mooncake_te);

  virtual ~MooncakeTransferEngineService() = default;

  virtual void OpenSession(google::protobuf::RpcController* controller,
                           const proto::SessionInfo* request,
                           proto::Status* response,
                           google::protobuf::Closure* done) override;

  virtual void CloseSession(google::protobuf::RpcController* controller,
                            const proto::SessionInfo* request,
                            proto::Status* response,
                            google::protobuf::Closure* done) override;

 private:
  MooncakeTransferEngine* mooncake_te_;
};

}  // namespace xllm
