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

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <folly/futures/Future.h>

#include <memory>
#include <string>
#include <vector>

#include "framework/xtensor/xtensor.h"
#include "runtime/forward_params.h"
#include "runtime/params_utils.h"
#include "worker.pb.h"

namespace xllm {

class CommChannel {
 public:
  CommChannel() = default;
  virtual ~CommChannel() = default;

  bool init_brpc(const std::string& server_address);

  virtual bool hello();

  virtual bool allocate_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual bool allocate_continuous_kv_cache(
      const std::vector<XTensor::Options>& options);

  virtual bool get_device_info(std::string& device_ip, uint16_t& port);

  virtual bool get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              int64_t& k_cache_id,
                              int64_t& v_cache_id);

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports);

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports);

  virtual bool init_model(const std::string& model_weights_path);

  virtual bool init_model_async(const std::string& model_weights_path,
                                folly::Promise<bool>& promise);

  virtual bool estimate_kv_cache_capacity(int64_t& available_memory,
                                          int64_t& total_memory);

  virtual bool pull_kv_blocks(const uint64_t src_cluster_id,
                              const std::string& src_addr,
                              const int64_t src_k_cache_id,
                              const int64_t src_v_cache_id,
                              const std::vector<uint64_t>& src_blocks,
                              const std::vector<uint64_t>& dst_blocks);

  virtual void execute_model_async(
      const std::vector<RawForwardInput>& inputs,
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  virtual bool process_group_test();

  virtual bool allocate_kv_cache_with_transfer(
      const uint64_t kv_cache_size,
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual void transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      folly::Promise<uint32_t>& promise);

  virtual void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  virtual void prefetch_from_storage(
      const std::atomic<bool>& flag,
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<uint32_t>>& success_cnt);

  virtual bool get_last_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  virtual bool get_active_activation_memory(int64_t& memory);

  virtual bool get_active_activation_memory_async(
      folly::Promise<int64_t>& promise);

 protected:
  bool execute_model_with_brpc(
      const std::vector<RawForwardInput>& inputs,
      folly::Promise<std::optional<RawForwardOutput>>& promise);

 private:
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::DistributeWorker_Stub> stub_;
};

class InitModelClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::Status response;
  brpc::Controller cntl;
  folly::Promise<bool> promise;
};

class ExecuteModelClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::ForwardOutput pb_output;
  brpc::Controller cntl;
  folly::Promise<std::optional<RawForwardOutput>> promise;
};

class TransferBlocksClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::TransferStatus response;
  brpc::Controller cntl;
  folly::Promise<uint32_t> promise;
};
}  // namespace xllm