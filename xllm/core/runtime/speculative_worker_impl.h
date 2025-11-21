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

#include "common/macros.h"
#include "framework/kv_cache/token_cache_allocator.h"
#if defined(USE_NPU)
#include "framework/kv_cache/spec_kv_cache_transfer.h"
#endif
#include "runtime/llm_worker_impl.h"
#include "runtime/options.h"

namespace xllm {

#if defined(USE_NPU)
using namespace llm_datadist;
#endif

class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options);

  ~SpeculativeWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override {
    // do nothing
    return true;
  };

  bool init_model(const std::string& model_weights_path) override;

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
      const uint64_t kv_cache_size,
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
  void prepare_work_before_execute(const BatchedForwardInputs& inputs,
                                   BatchedForwardInputs& new_inputs) override;

  std::optional<ForwardOutput> step(
      const BatchedForwardInputs& inputs) override;

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

 private:
  // When enable DP, inputs sometimes be empty but model need to execute.
  std::optional<ForwardOutput> step_empty(ForwardInput& input);
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input);
  std::optional<ForwardOutput> step_decode(const ForwardInput& input);

  folly::SemiFuture<std::optional<ForwardOutput>> get_output_async(
      LLMWorkerImpl* impl,
      const ForwardInput& input) {
    BatchedForwardInputs batch_inputs;
    batch_inputs.micro_inputs.push_back(input);
    batch_inputs.concated_sampling_params = input.sampling_params;
    return impl->step_async(batch_inputs);
  };

  // prepare first step inputs of draft model at Prefill phase.
  void prepare_first_prefill_input(const ForwardInput& input,
                                   ForwardInput& draft_input);

  // prepare first step inputs of draft model at Decode phase.
  void prepare_first_decode_input(const ForwardInput& input,
                                  ForwardInput& draft_input,
                                  const SampleOutput val_output);

  // prepare next step inputs of draft model.
  void prepare_draft_input(const ForwardInput& input,
                           ForwardInput& draft_input,
                           const int64_t offset);

  // prepare inputs of target model for validation at Decode phase.
  void prepare_validate_input(const ForwardInput& input,
                              ForwardInput& validate_input,
                              std::vector<std::vector<int32_t>>& draft_tokens);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        std::vector<std::vector<int32_t>>& draft_tokens,
                        const ForwardOutput& target_output);

 private:
  int32_t embedding_size_ = 0;

  std::unique_ptr<LLMWorkerImpl> impl_;
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  std::shared_ptr<TokenCacheAllocator> token_allocator_;
#if defined(USE_NPU)
  std::shared_ptr<SpecKVCacheTransfer> kv_cache_transfer_;
#endif
};
}  // namespace xllm
