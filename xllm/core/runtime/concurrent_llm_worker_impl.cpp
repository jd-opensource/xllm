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

#include "concurrent_llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

ConcurrentLLMWorkerImpl::ConcurrentLLMWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options),
      max_concurrency_(FLAGS_llm_worker_max_concurrency) {
  CHECK_GT(max_concurrency_, 0)
      << "llm_worker_max_concurrency must be greater than 0";

  device_.set_device();

  // Create independent step_threadpool_ dedicated to parallel execution of
  // step() Use schedule() to assign tasks, letting ThreadPool automatically
  // select idle threads
  step_threadpool_ = std::make_unique<ThreadPool>(
      max_concurrency_, [this]() mutable { device_.set_device(); });

  LOG(INFO) << "ConcurrentLLMWorkerImpl: Created step_threadpool_ with "
            << max_concurrency_ << " threads for parallel step execution";

#if defined(USE_CUDA)
  // initialize flashinfer workspace
  layer::FlashinferWorkspace::get_instance().initialize(device_);
#endif
}

bool ConcurrentLLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Create multiple model instances
  model_instances_.reserve(max_concurrency_);
  executor_instances_.reserve(max_concurrency_);
  execute_streams_.reserve(max_concurrency_);
  context_instances_.reserve(max_concurrency_);

  for (int32_t i = 0; i < max_concurrency_; ++i) {
    // Create corresponding execute stream
    auto stream = device_.get_stream_from_pool();
    execute_streams_.push_back(std::move(stream));

    auto stream_guard = execute_streams_[i]->set_stream_guard();
    // Create independent ModelContext for each model instance
    // Use constructor to create new context, ensuring each instance has
    // independent context For NPU, this creates a new atb::Context
    ModelContext instance_context(context.get_parallel_args(),
                                  context.get_model_args(),
                                  context.get_quant_args(),
                                  context.get_tensor_options());
    context_instances_.push_back(std::move(instance_context));

    // Create model instance using the corresponding context
    auto model_instance = create_llm_model(context_instances_[i]);
    CHECK(model_instance != nullptr) << "Failed to create model instance " << i;
    model_instances_.push_back(std::move(model_instance));

    // Create corresponding executor using the corresponding context
    auto executor =
        std::make_unique<Executor>(model_instances_[i].get(),
                                   context_instances_[i].get_model_args(),
                                   device_,
                                   options_);
    executor_instances_.push_back(std::move(executor));

    LOG(INFO) << "Created model instance " << i
              << " with executor, execute stream and context";
  }

  // For compatibility with base class interface, set base class's model_ and
  // model_executor_ to point to the first instance Note: Need to access base
  // class's protected members model_ and model_executor_ Use reset() to set
  // pointers, but note: ownership of model_ actually belongs to
  // model_instances_[0]. In destructor, need to release model_ and
  // model_executor_ first to avoid double deletion
  model_.reset(model_instances_[0].get());
  model_executor_.reset(executor_instances_[0].get());

  // Complete other initialization (EPLB, BeamSearcher, etc.)
  // Note: These are members of base class LLMWorkerImpl, can be accessed
  // directly
  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    // Use base class's protected member beam_searcher_
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  return true;
}

void ConcurrentLLMWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!model_instances_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  // Save model weights path to create new loaders for other instances
  std::string model_weights_path = loader->model_weights_path();

  // Load weights for the first model instance (using the original loader)
  model_instances_[0]->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  // Create new loaders and load weights for other model instances
  for (size_t i = 1; i < model_instances_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    model_instances_[i]->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << model_instances_.size()
            << " model instances";
}

void ConcurrentLLMWorkerImpl::allocate_instance_id_for_current_thread() {
  std::thread::id current_thread_id = std::this_thread::get_id();

  // Lock to protect the allocation process
  std::lock_guard<std::mutex> lock(allocation_mutex_);

  // Check if current thread is already in the map (may have been allocated by
  // other tasks)
  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it != thread_id_to_instance_id_.end()) {
    return;
  }

  // Select the smallest unallocated instance id
  size_t instance_id = SIZE_MAX;
  size_t stream_num = static_cast<size_t>(max_concurrency_);
  for (size_t i = 0; i < stream_num; ++i) {
    if (allocated_instance_ids_.find(i) == allocated_instance_ids_.end()) {
      instance_id = i;
      break;
    }
  }

  CHECK_NE(instance_id, SIZE_MAX)
      << "No available instance id, all " << max_concurrency_
      << " instance ids are allocated";

  // Establish mapping relationship
  thread_id_to_instance_id_[current_thread_id] = instance_id;
  allocated_instance_ids_.insert(instance_id);

  LOG(INFO) << "Allocated instance_id " << instance_id << " for thread "
            << current_thread_id;
}

void ConcurrentLLMWorkerImpl::get_thread_model_instance(
    CausalLM*& model,
    Executor*& executor,
    Stream*& execute_stream,
    ModelContext*& context) {
  std::thread::id current_thread_id = std::this_thread::get_id();

  // If current thread hasn't been allocated an instance id yet, allocate it
  // first
  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it == thread_id_to_instance_id_.end()) {
    allocate_instance_id_for_current_thread();
    it = thread_id_to_instance_id_.find(current_thread_id);
  }

  CHECK(it != thread_id_to_instance_id_.end())
      << "Failed to find instance id for thread " << current_thread_id;
  size_t instance_id = it->second;
  // LOG(INFO) << "get_thread_model_instance: thread " << current_thread_id
  // << " allocated instance_id " << instance_id;

  CHECK_LT(instance_id, model_instances_.size())
      << "Thread model index " << instance_id
      << " exceeds model instances size " << model_instances_.size();

  model = model_instances_[instance_id].get();
  executor = executor_instances_[instance_id].get();
  execute_stream = execute_streams_[instance_id].get();
  context = &context_instances_[instance_id];
}

folly::SemiFuture<std::optional<ForwardOutput>>
ConcurrentLLMWorkerImpl::step_async(const ForwardInput& input) {
  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();

  // Use schedule() to assign tasks, letting ThreadPool automatically select
  // idle threads The logic for allocating instance_id happens when the task
  // executes (see lambda below)
  step_threadpool_->schedule([this,
                              input = std::move(input_on_device),
                              promise = std::move(promise)]() mutable {
    // When the task executes, if the current thread hasn't been allocated an
    // instance id yet, allocate it The allocation logic will lock, select the
    // smallest unallocated instance id, establish a mapping from thread id to
    // instance id Once allocation is complete, the mapping relationship is
    // saved in thread_id_to_instance_id_. This way, multiple threads complete
    // allocation after executing once

    // Handle hierarchy_kv_cache_transfer if needed (from base class logic)
    if (hierarchy_kv_cache_transfer_ != nullptr) {
      hierarchy_kv_cache_transfer_->set_layer_synchronizer(input.input_params);
    }

    // Call step() using the model instance corresponding to the current thread
    const auto output = this->step(input);

    // Handle enable_schedule_overlap logic (if needed)
    if (!enable_schedule_overlap()) {
      promise.setValue(output);
    } else {
      if (last_step_output_valid_ && !input.input_params.empty_kv_cache) {
        // replace step i model input with true output of step i-1
        input = update_input_by_last_step_output(input);
      }

      const auto output_overlap = this->step(input);
      if (output_overlap.has_value()) {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          update_last_step_output(output_overlap);
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          update_last_step_output(output_overlap);
        }
      } else {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          last_step_output_valid_ = false;
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          last_step_output_valid_ = false;
        }
      }
      promise.setValue(output_overlap);
    }
  });
  return future;
}

std::optional<ForwardOutput> ConcurrentLLMWorkerImpl::step(
    const ForwardInput& input) {
  Timer timer;
  auto& sampling_params = input.sampling_params;

  // Get the model, executor, stream and context corresponding to the current
  // thread
  CausalLM* model = nullptr;
  Executor* executor = nullptr;
  Stream* execute_stream = nullptr;
  ModelContext* context = nullptr;
  get_thread_model_instance(model, executor, execute_stream, context);

  c10::StreamGuard stream_guard = execute_stream->set_stream_guard();

  std::vector<folly::SemiFuture<bool>> futures;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context->get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context->get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(input.eplb_info);
  }

  // Use the executor and model corresponding to the thread
  auto hidden_states = executor->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model->logits(hidden_states, sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    // Synchronize the current thread's stream (if using independent stream)
    if (execute_stream != nullptr) {
      execute_stream->synchronize();
    } else {
      device_.synchronize_default_stream();
    }

    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
        input.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(input.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  if (options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() && !is_spec_draft_) {
      output.sample_output.embeddings = hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  // Synchronize the current thread's stream (if using independent stream)
  if (execute_stream != nullptr) {
    execute_stream->synchronize();
  } else {
    device_.synchronize_default_stream();
  }

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

void ConcurrentLLMWorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  // Implement the same logic as the base class because the base class's method
  // is private
  if (output.has_value()) {
    if (output.value().sample_output.next_tokens.defined()) {
      last_step_output_ = std::move(output.value());
      last_step_output_valid_ = true;
    } else {
      if (FLAGS_enable_eplb) {
        last_step_output_ = std::move(output.value());
      }
      last_step_output_valid_ = false;
    }
  } else {
    last_step_output_valid_ = false;
  }
}

}  // namespace xllm
