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

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecWorkPipeline> create_pipeline(
      RecPipelineType type,
      RecWorkerImpl& worker);

  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int64_t>& input_indices);

  std::unique_ptr<RecWorkPipeline> work_pipeline_;

  RecModelKind rec_model_kind_ = RecModelKind::kNone;
};

}  // namespace xllm
