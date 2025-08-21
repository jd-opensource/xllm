/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "distributed_runtime/dist_manager.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/eplb/eplb_manager.h"
#include "framework/eplb/eplb_policy.h"
#include "framework/quant_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/engine.h"
#include "util/threadpool.h"
#include "worker.h"
#include "worker_client.h"
#include "xservice_client.h"

namespace xllm {

class LLMEngine : public Engine {
 public:
  // create an engine with the given devices
  LLMEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  virtual ~LLMEngine() = default;

  ForwardOutput step(std::vector<Batch>& batch) override;

  const runtime::Options& options() const { return options_; }

  bool init() override;

  void update_last_step_result(std::vector<Batch>& batch) override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const override;

  // P/D
  bool pull_kv_blocks(const int32_t src_dp_size,
                      const int32_t src_dp_rank,
                      const std::vector<uint64_t>& src_cluster_ids,
                      const std::vector<std::string>& src_addrs,
                      const std::vector<int64_t>& src_k_cache_ids,
                      const std::vector<int64_t>& src_v_cache_ids,
                      const std::vector<uint64_t>& src_blocks,
                      const int32_t dst_dp_rank,
                      const std::vector<uint64_t>& dst_blocks) override;

  std::vector<folly::SemiFuture<uint32_t>> load_kv_blocks_from_store_async(
      const uint32_t dp_rank,
      const std::vector<CacheBlockInfo>& cache_block_info) override;

  void get_device_info(std::vector<std::string>& device_ips,
                       std::vector<uint16_t>& ports) override;

  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<std::string>& addrs,
                      std::vector<int64_t>& k_cache_ids,
                      std::vector<int64_t>& v_cache_ids) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<std::string>& device_ips,
                    const std::vector<uint16_t>& ports,
                    const int32_t src_dp_size) override;

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<std::string>& device_ips,
                      const std::vector<uint16_t>& ports,
                      const int32_t dp_size) override;

  std::shared_ptr<DistManager> get_dist_manager() { return dist_manager_; };

 private:
  friend class SpeculativeEngine;
  // setup workers internal
  void setup_workers(const runtime::Options& options);
  bool init_model();
  Engine::KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap);
  bool allocate_continuous_kv_cache(
      const Engine::KVCacheCapacity& kv_cache_cap);

 protected:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // quantization args
  QuantArgs quant_args_;

  // worker client which is used for call worker
  // The reason for adding a worker client is to unify the
  // access code for both local and remote workers, thereby
  // introducing an additional worker_client abstraction.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // config for kv cache
  int64_t n_local_kv_heads_ = 0;
  int64_t n_local_q_heads_ = 0;
  int64_t head_dim_ = 0;

  torch::Tensor expert_load_data_;
  // For multi-node serving
  // engine brpc server, all workers connect to engine_server_,
  // engine_server_ will send a UniqueId for workers to
  // create process group. And workers send worker brpc server
  // address to engine, engine will create WorkerClient for each worker.
  // Engine call workers to step via these WorkerClients.
  std::shared_ptr<DistManager> dist_manager_ = nullptr;

  std::unique_ptr<EplbManager> eplb_manager_ = nullptr;
  void process_eplb_data(
      const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results,
      int32_t worker_clients_num);

  ThreadPool threadpool_;
};

}  // namespace xllm
