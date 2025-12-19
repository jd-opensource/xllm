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

#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "llm_worker_impl.h"
#include "platform/device.h"
#include "platform/stream.h"
#include "util/threadpool.h"

namespace xllm {

// ConcurrentLLMWorkerImpl: LLM Worker supporting multi-stream parallel
// execution Inherits from LLMWorkerImpl, adds support for multiple model
// instances and execute stream pool
class ConcurrentLLMWorkerImpl : public LLMWorkerImpl {
 public:
  // execute_stream_num: execution parallelism, determines the number of model
  // instances and execute streams
  explicit ConcurrentLLMWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options);

  ~ConcurrentLLMWorkerImpl() override {
    // Release model_ and model_executor_ in destructor to avoid double deletion
    // Ownership actually belongs to model_instances_[0] and
    // executor_instances_[0]
    model_.release();
    model_executor_.release();
  }

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  // Override load_model to load weights for all model instances
  void load_model(std::unique_ptr<ModelLoader> loader) override;

  // Override step_async to support multi-threaded parallel execution
  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 private:
  // Execution parallelism (number of model instances and execute streams)
  uint32_t max_concurrency_;

  // Multiple model instances (one per stream)
  // Note: Multiple model instances are needed because many model forward()
  // methods don't support parallel execution Multiple threads concurrently
  // calling the same model's forward() will cause data races and undefined
  // behavior Therefore, we need to create independent model instances for each
  // parallel execution thread
  std::vector<std::unique_ptr<CausalLM>> model_instances_;

  // Multiple executor instances (one per stream, corresponding to
  // model_instances_)
  std::vector<std::unique_ptr<Executor>> executor_instances_;

  // Execute stream pool (one stream per model instance)
  std::vector<std::unique_ptr<Stream>> execute_streams_;

  // Multiple ModelContext instances (one per model instance)
  // Each context instance contains independent model args, parallel args, etc.
  std::vector<ModelContext> context_instances_;

  // Independent ThreadPool dedicated to parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  // Mapping from thread id to instance id (used to find the instance id for the
  // current thread)
  std::unordered_map<std::thread::id, size_t> thread_id_to_instance_id_;

  // Set of allocated instance ids (used to select the smallest unallocated
  // instance id)
  std::set<size_t> allocated_instance_ids_;

  // Mutex protecting the allocation process
  std::mutex allocation_mutex_;

  // Helper method: Get the corresponding model, executor, stream and context
  // based on thread ID
  void get_thread_model_instance(CausalLM*& model,
                                 Executor*& executor,
                                 Stream*& execute_stream,
                                 ModelContext*& context);

  // Allocate instance id for the current thread (thread-safe)
  void allocate_instance_id_for_current_thread();

  // Update last_step_output (because the base class's update_last_step_output
  // is private)
  void update_last_step_output(const std::optional<ForwardOutput>& output);
};

}  // namespace xllm
