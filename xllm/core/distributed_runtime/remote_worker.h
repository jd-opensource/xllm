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
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "common/macros.h"
#include "framework/model/causal_lm.h"
#include "framework/model/embedding_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/executor.h"
#include "runtime/forward_params.h"
#include "runtime/worker_client.h"
#include "util/threadpool.h"
#include "worker.pb.h"

namespace xllm {

class RemoteWorker : public WorkerClient {
 public:
  explicit RemoteWorker(int32_t global_rank,
                        const std::string& server_address,
                        const torch::Device& d);
  virtual ~RemoteWorker() = default;

  bool wait_for_server_ready(const std::string& server_address);

  virtual bool init_model(const std::string& model_weights_path) override;

  virtual std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  virtual bool allocate_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;

  virtual bool allocate_continuous_kv_cache(
      const std::vector<XTensor::Options>& options) override;

  virtual void get_device_info(std::string& device_ip, uint16_t& port);

  virtual void get_cache_info(uint64_t& cluster_id,
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

  virtual bool pull_kv_blocks(const uint64_t src_cluster_id,
                              const std::string& src_addr,
                              const int64_t src_k_cache_id,
                              const int64_t src_v_cache_id,
                              const std::vector<uint64_t>& src_blocks,
                              const std::vector<uint64_t>& dst_blocks);

  // prepare input request
  virtual ForwardInput prepare_inputs(Batch& batch) override;

  virtual std::optional<ForwardOutput> step(
      const ForwardInput& inputs) override;

  virtual folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path) override;

  virtual folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async() override;

  virtual folly::SemiFuture<bool> allocate_kv_cache_async(
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;

  virtual folly::SemiFuture<bool> allocate_continuous_kv_cache_async(
      const std::vector<XTensor::Options>& options) override;

  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      const uint64_t kv_cache_size,
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const int64_t src_k_cache_id,
      const int64_t src_v_cache_id,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks);

  virtual folly::SemiFuture<uint32_t> load_kv_blocks_from_store_async(
      const std::vector<CacheBlockInfo> cache_block_info);

  // Run the model and return the output.
  virtual folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs) override;

  virtual folly::SemiFuture<std::optional<RawForwardOutput>> step_async(
      const RawForwardInput& inputs) override;

  virtual folly::SemiFuture<folly::Unit> process_group_test_async() override;

  virtual const torch::Device& device() const override;

  folly::SemiFuture<std::optional<RawForwardOutput>>
  get_last_step_result_async();

  virtual int64_t get_active_activation_memory() override;

  virtual folly::SemiFuture<int64_t> get_active_activation_memory_async()
      override;

 private:
  DISALLOW_COPY_AND_ASSIGN(RemoteWorker);

 private:
  int32_t global_rank_;

  // brpc connection resource
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::DistributeWorker_Stub> stub_;

  ThreadPool threadpool_;
  // general working thread
  // do some overlap work with model execute
  ThreadPool general_threadpool_{5};
  const torch::Device device_;
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

class LoadKVCacheFromStoreClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::StoreResponse response;
  brpc::Controller cntl;
  folly::Promise<uint32_t> promise;
};

}  // namespace xllm
