/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/embedding_cache.h"
#if defined(USE_NPU)
#include "framework/kv_cache/spec_kv_cache_transfer.h"
#endif
#include "runtime/speculative_worker_impl.h"

namespace xllm {

#if defined(USE_NPU)
using namespace llm_datadist;
#endif

// MTP (Multi-Token Prediction) speculative worker.
// Uses a draft model to generate proposals, then validates with target model.
// Eagle3WorkerImpl inherits from this class.
class MTPWorkerImpl : public SpeculativeWorkerImpl {
 public:
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~MTPWorkerImpl() override = default;

 protected:
  // For derived classes (e.g. Eagle3WorkerImpl) that need custom options for
  // target and draft models. `options` is passed to WorkerImpl (preserves
  // enable_schedule_overlap etc.), `target_options` / `draft_options` are used
  // to create the respective workers.
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                const runtime::Options& target_options,
                const runtime::Options& draft_options,
                bool enable_opt_validate_probs = false);

 public:
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed) override;

  bool allocate_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const uint64_t kv_cache_size,
      const std::vector<std::vector<int64_t>>& kv_cache_shape) override;
#endif

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

  virtual SampleOutput validate(const SamplingParameters& sampling_params,
                                const std::vector<ForwardOutput>& draft_outputs,
                                const ForwardOutput& target_output);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output);

  // PD separation: placeholder size for empty embedding slot. Default: 1x
  // hidden_size. Eagle3 overrides to 3 * target_hidden_size.
  virtual int64_t get_embedding_placeholder_size();

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const ForwardInput& inputs,
                              ForwardInput& prefill_inputs);

  // prepare inputs for draft model at Decode phase.
  void prepare_draft_inputs(const ForwardInput& inputs,
                            ForwardInput& draft_inputs,
                            const int64_t offset,
                            const torch::Device device);

 protected:
  // Draft model worker
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Embedding cache for speculative decoding
  std::shared_ptr<EmbeddingCache> embedding_cache_;

  // Whether to use optimized draft_probs handling in validation.
  bool enable_opt_validate_probs_ = false;

#if defined(USE_NPU)
  std::shared_ptr<SpecKVCacheTransfer> kv_cache_transfer_;
#endif
};
}  // namespace xllm
