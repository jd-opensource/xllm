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

#include <vector>

#include "common/macros.h"
#include "framework/sampling/rejection_sampler.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/options.h"

namespace xllm {

// Base class for all speculative decoding workers.
// Provides common logic: target model management, step dispatch, and
// sampling parameter updates. Subclasses implement algorithm-specific
// draft generation and validation (MTP, Eagle3, suffix, etc.).
class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  ~SpeculativeWorkerImpl() override = default;

 protected:
  // `options` is passed to WorkerImpl (preserves enable_schedule_overlap etc.),
  // `target_options` is used to create impl_ (target model worker).
  // Each algorithm subclass decides its own target_options.
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        const runtime::Options& target_options);

 public:
  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override {
    // do nothing
    return true;
  };

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  int32_t master_status) override;

  void get_device_info(std::string& device_ip, uint16_t& port) override {
    impl_->get_device_info(device_ip, port);
  };

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<std::string>& device_ips,
                    const std::vector<uint16_t>& ports) override {
    return impl_->link_cluster(cluster_ids, addrs, device_ips, ports);
  };

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<std::string>& device_ips,
                      const std::vector<uint16_t>& ports) override {
    return impl_->unlink_cluster(cluster_ids, addrs, device_ips, ports);
  };

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override {
    return impl_->estimate_kv_cache_capacity();
  };

  // allocate kv cache. blocking call
  bool allocate_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;
#endif

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      int64_t& k_cache_id,
                      int64_t& v_cache_id) override {
    impl_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
  };

  // prepare input for execution
  ForwardInput prepare_inputs(Batch& batch) override {
    return impl_->prepare_inputs(batch);
  };

  // prepare work before model execution
  void prepare_work_before_execute(const ForwardInput& input,
                                   ForwardInput& new_input) override;

  // Common step dispatch: prefill / decode / empty
  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;

  folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const int64_t src_k_cache_id,
      const int64_t src_v_cache_id,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks) override {
    return impl_->pull_kv_blocks_async(src_cluster_id,
                                       src_addr,
                                       src_k_cache_id,
                                       src_v_cache_id,
                                       src_blocks,
                                       dst_blocks);
  };

 protected:
  // Algorithm-specific virtual methods for subclasses to implement
  virtual std::optional<ForwardOutput> step_prefill(
      const ForwardInput& input) = 0;
  virtual std::optional<ForwardOutput> step_decode(
      const ForwardInput& inputs) = 0;
  virtual std::optional<ForwardOutput> step_empty(
      const ForwardInput& inputs) = 0;

  // Common helper: update sampling params for validation
  void update_sampling_params(SamplingParameters& sampling_params,
                              const int32_t num_val_tokens,
                              const int32_t total_num_val_tokens);

  // prepare inputs for target model at Decode phase (validation).
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs);

 protected:
  // Target model worker
  std::unique_ptr<LLMWorkerImpl> impl_;

  // performance debug for fixing the speculative acceptance rate
  // NOTE: This is for performance debugging only, it will
  // influence the model accuracy and should not be used in production.
  std::shared_ptr<RejectionSamplerRateController> rate_controller_;

  bool enable_fused_kernel_ = false;
  int32_t embedding_size_ = 0;
};
}  // namespace xllm
