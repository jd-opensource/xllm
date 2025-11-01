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

#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "framework/batch/batch.h"
#include "framework/model_context.h"
#include "framework/sampling/valid_path_filter.h"
#include "platform/stream.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"
#include "worker_impl.h"

namespace xllm {

// Rec specific worker implementation
class RecWorkerImpl : public WorkerImpl {
 public:
  RecWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  // Override init_model for Rec specific implementation
  bool init_model(const std::string& model_weights_path) override;

  // Override init_model with ModelContext for Rec specific implementation
  bool init_model(ModelContext& context) override;

  // Override step for Rec specific implementation
  std::optional<ForwardOutput> step(
      const BatchedForwardInputs& inputs) override;

  // Override prepare_inputs for Rec specific implementation
  ForwardInput prepare_inputs(Batch& batch) override;

 private:
  // Helper method for filter mask preparation (placeholder for future
  // implementation)
  torch::Tensor prepare_filter_mask(
      const std::vector<std::vector<int32_t>>& generated_tokens);

  // Async filter mask preparation with overlap
  std::future<torch::Tensor> prepare_filter_mask_async(
      const std::vector<std::vector<int32_t>>& generated_tokens);

  // Stream for H2D memory copy operations
  std::unique_ptr<Stream> filter_mask_stream_;

  // ThreadPool for async operations
  std::shared_ptr<ThreadPool> thread_pool_;

  // ValidPathFilter for beam search filtering
  std::unique_ptr<ValidPathFilter> valid_path_filter_;

  // BeamSearcher for beam search functionality
  std::unique_ptr<BeamSearcher> beam_searcher_;
};

}  // namespace xllm