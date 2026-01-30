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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/rec_model_utils.h"
#include "runtime/llm_worker_impl.h"
#include "util/threadpool.h"

namespace xllm {

class RecWorkerImpl : public LLMWorkerImpl {
 public:
  RecWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  bool init_model(ModelContext& context) override;

  ForwardInput prepare_inputs(Batch& batch) override;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 protected:
  std::shared_ptr<ThreadPool> input_builder_thread_pool_;

 private:
  class RecWorkPipeline {
   public:
    virtual ~RecWorkPipeline() = default;

    virtual bool create_model(RecWorkerImpl& worker, ModelContext& context) = 0;

    virtual ForwardInput prepare_inputs(Batch& batch) = 0;

    virtual void prepare_work_before_execute(
        const ForwardInput& inputs,
        ForwardInput& processed_inputs) = 0;

    virtual std::optional<ForwardOutput> step(const ForwardInput& input) = 0;
  };

  class LlmRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class OneRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit OneRecWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class LlmRecWithMmDataWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWithMmDataWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class LlmRecMultiRoundPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecMultiRoundPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    // Beam search related tensors
    struct BeamSearchTensors {
      torch::Tensor sequence_group;   // [batch_size, beam_width, total_rounds]
      torch::Tensor acc_logprob;      // [num_seq, 1]
      torch::Tensor out_log_probs;    // [num_seq, 1]
      torch::Tensor out_token_ids;    // [num_seq, 1]
      torch::Tensor out_token_index;  // [num_seq, 1]
      torch::Tensor out_beam_count_prefix_sums;  // [num_seq, 1]
      torch::Tensor out_seqgroup;  // [batch_size, beam_width, total_rounds]
    };

    // Prepare beam search tensors
    BeamSearchTensors prepare_beam_search_tensors(int32_t batch_size,
                                                  int32_t beam_width,
                                                  int32_t total_rounds,
                                                  const torch::Device& device);

    // Execute beam search kernel
    void execute_beam_search(const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             BeamSearchTensors& beam_tensors,
                             int32_t round,
                             int32_t batch_size);

    // Execute cache select kernel
    void execute_cache_select(const BeamSearchTensors& beam_tensors,
                              ForwardInput& input,
                              int32_t round,
                              int32_t beam_width,
                              int32_t layer_num);

    // Build final output from beam search results
    void build_final_output(const torch::Tensor& logits,
                            const SampleOutput& sample_output,
                            const SamplingParameters& sampling_params,
                            const BeamSearchTensors& beam_tensors,
                            ForwardOutput& output);

    // Structure to hold async computation results for next round input
    struct NextRoundInputResults {
#if defined(USE_NPU)
// TODO: implement NextRoundInputResults for NPU
#elif defined(USE_CUDA)
      torch::Tensor paged_kv_indices;
      torch::Tensor paged_kv_indptr;
      torch::Tensor paged_kv_last_page_len;
#endif
    };

    // Compute next round input asynchronously (can overlap with GPU execution)
    folly::SemiFuture<NextRoundInputResults> compute_next_round_input_async(
        const torch::Tensor& kv_seq_lens,
        int32_t current_step,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step);

    // Apply async result to prepare decode input for current round
    void prepare_input_for_current_round(ForwardInput& input,
                                         const NextRoundInputResults& results,
                                         int32_t round,
                                         const torch::Tensor& top_tokens,
                                         const BeamSearchTensors& beam_tensors);

    // Consume async result for current round and schedule async computation for
    // next round.
    void prepare_round_input_and_schedule_next(
        ForwardInput& input,
        int32_t round,
        int32_t total_rounds,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step,
        const torch::Tensor& top_tokens,
        const BeamSearchTensors& beam_tensors,
        std::optional<folly::SemiFuture<NextRoundInputResults>>&
            next_round_async_result);

    void allocate_kv_caches_related();
    void prepare_kv_caches_related_for_input(const ForwardInput& inputs,
                                             ForwardInput& processed_inputs);

    struct FullKvCacheOffsets {
      explicit FullKvCacheOffsets(
          LlmRecMultiRoundPipeline* multi_round_pipeline);
#if defined(USE_NPU)
// TODO: implement FullKvCacheOffsets for NPU
#elif defined(USE_CUDA)
      torch::Tensor full_kv_offsets;
      torch::Tensor full_kv_mask;
      torch::Tensor full_kv_indices;
      torch::Tensor unshared_offsets;
      torch::Tensor max_decode_step_ids;
#endif
    };
    std::unique_ptr<FullKvCacheOffsets> full_kv_cache_offsets_;

    std::vector<torch::Tensor> cached_full_k_caches_;
    std::vector<torch::Tensor> cached_full_v_caches_;
    torch::Tensor cached_naive_block_table_;
    torch::Tensor cached_current_round_tensor_;

    RecWorkerImpl& worker_;

    // for async scheduler
    ThreadPool threadpool_;

    int32_t max_seqs_per_batch_{worker_.options_.max_seqs_per_batch()};
    int32_t max_token_per_req_{worker_.options_.max_token_per_req()};
    int32_t max_tokens_per_batch_{worker_.options_.max_seqs_per_batch() *
                                  worker_.options_.max_token_per_req()};
    int32_t beam_width_{worker_.options_.beam_width()};
  };

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecWorkPipeline> create_pipeline(
      RecPipelineType type,
      RecWorkerImpl& worker);

  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int64_t>& input_indices);

  void prepare_multi_modal_data(ForwardInput& processed_inputs);

  std::unique_ptr<RecWorkPipeline> work_pipeline_;

  RecModelKind rec_model_kind_ = RecModelKind::kNone;
};

}  // namespace xllm
