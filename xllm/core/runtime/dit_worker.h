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

#include <thread>

#include "dit_executor.h"
#include "framework/model_context.h"
#include "options.h"
#include "util/threadpool.h"

namespace xllm {

class DiTWorker {
 public:
  DiTWorker(const ParallelArgs& parallel_args,
            const torch::Device& device,
            const runtime::Options& options);

  ~DiTWorker() = default;

  // initialize model, cache manager. blocking call
  bool init_model(const std::string& model_weights_path);

  std::optional<DiTForwardOutput> step(const DiTForwardInput& inputs);

  folly::SemiFuture<folly::Unit> process_group_test_async();

  // prepare input for execution
  DiTForwardInput prepare_inputs(DiTBatch& batch);

  int64_t get_active_activation_memory();

 private:
  runtime::Options options_;

  std::unique_ptr<DiTModel> dit_model_;

  std::unique_ptr<DiTExecutor> dit_model_executor_;

  torch::Device device_;

  torch::ScalarType dtype_;

  // model context, includes model args, parallel args and date type etc.
  mutable ModelContext context_;

  ParallelArgs parallel_args_;

  ThreadPool threadpool_;
};

}  // namespace xllm