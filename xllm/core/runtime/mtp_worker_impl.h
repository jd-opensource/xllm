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
#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"
#endif
#include "runtime/spec_input_builder.h"
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
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;
  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

  void fill_validate_input_from_draft_outputs(
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input);
  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input);

  virtual SampleOutput validate(const SamplingParameters& sampling_params,
                                const std::vector<ForwardOutput>& draft_outputs,
                                const ForwardOutput& target_output);

  // Hook for algorithm-specific draft output post-processing during decode.
  // Default MTP behavior always compresses probs for cache storage.
  virtual void process_draft_sample_output(SampleOutput& sample_output);

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

  // Prepare target validate input from cached target context.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               const specBuilder::DecodeCpuView& decode_view,
                               ForwardInput& validate_inputs);

  // prepare inputs for draft model at Decode phase.
  void prepare_draft_inputs(const ForwardInput& inputs,
                            const specBuilder::DecodeCpuView& decode_view,
                            ForwardInput& draft_inputs,
                            int32_t position_offset);
  void build_decode_step_view(
      const ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states,
      specBuilder::DecodeCpuView& decode_view) const;

  // Build draft-side input from cached target context at decode step start.
  void prepare_draft_extend_inputs(
      const ForwardInput& base_input,
      const specBuilder::DecodeCpuView& decode_view,
      const std::vector<EmbeddingCache::DecodeState>& last_states,
      ForwardInput& extend_input);

  void write_target_context_to_cache(const ForwardInput& input,
                                     const SampleOutput& validate_output);
  void record_validate_metrics(const SampleOutput& validate_output) const;

 protected:
  // Draft model worker
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Embedding cache for speculative decoding
  std::shared_ptr<EmbeddingCache> embedding_cache_;

  // Whether validation directly uses selected-only draft_probs [B, S].
  // If false, selected-only cache values are restored to dense [B, S, V].
  bool enable_opt_validate_probs_ = false;

#if defined(USE_NPU)
  std::shared_ptr<SpecKVCacheTransfer> kv_cache_transfer_;
#endif
};
}  // namespace xllm
