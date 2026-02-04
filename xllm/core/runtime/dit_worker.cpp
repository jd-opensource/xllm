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

#include "dit_worker.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/dit_model_loader.h"
#include "core/platform/device.h"
#include "framework/dit_cache/dit_cache.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
DiTCacheConfig parse_dit_cache_from_flags() {
  DiTCacheConfig cache_config;
  if (FLAGS_dit_cache_policy == "FBCache") {
    cache_config.selected_policy = PolicyType::FBCache;
    cache_config.fbcache.warmup_steps = FLAGS_dit_cache_warmup_steps;
    cache_config.fbcache.residual_diff_threshold =
        FLAGS_dit_cache_residual_diff_threshold;
  } else if (FLAGS_dit_cache_policy == "TaylorSeer") {
    cache_config.selected_policy = PolicyType::TaylorSeer;
    cache_config.taylorseer.n_derivatives = FLAGS_dit_cache_n_derivatives;
    cache_config.taylorseer.skip_interval_steps =
        FLAGS_dit_cache_skip_interval_steps;
    cache_config.taylorseer.warmup_steps = FLAGS_dit_cache_warmup_steps;
  } else if (FLAGS_dit_cache_policy == "FBCacheTaylorSeer") {
    cache_config.selected_policy = PolicyType::FBCacheTaylorSeer;
    cache_config.fbcachetaylorseer.n_derivatives =
        FLAGS_dit_cache_n_derivatives;
    cache_config.fbcachetaylorseer.warmup_steps = FLAGS_dit_cache_warmup_steps;
    cache_config.fbcachetaylorseer.residual_diff_threshold =
        FLAGS_dit_cache_residual_diff_threshold;
  } else if (FLAGS_dit_cache_policy == "None") {
    cache_config.selected_policy = PolicyType::TaylorSeer;
  }
  return cache_config;
}
}  // namespace

DiTWorker::DiTWorker(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const runtime::Options& options)
    : device_(device), options_(options), parallel_args_(parallel_args) {
  device_.set_device();
}

bool DiTWorker::init_model(const std::string& model_weights_path) {
  CHECK(dit_model_ == nullptr) << "Model is already initialized.";

  auto loader = std::make_unique<DiTModelLoader>(model_weights_path);
  dtype_ = util::parse_dtype(loader->get_torch_dtype(), device_);

  auto tensor_options = torch::dtype(dtype_).device(device_);
  context_ = DiTModelContext(parallel_args_,
                             std::move(loader->get_model_args()),
                             std::move(loader->get_quant_args()),
                             tensor_options,
                             options_.model_id());

  dit_model_ = create_dit_model(context_);
  CHECK(dit_model_ != nullptr) << "Failed to create model.";
  dit_model_->load_model(std::move(loader));

  dit_model_executor_ =
      std::make_unique<DiTExecutor>(dit_model_.get(), options_);

  DiTCacheConfig cache_config = parse_dit_cache_from_flags();
  DiTCache::get_instance().init(cache_config);

  return true;
}

folly::SemiFuture<bool> DiTWorker::init_model_async(
    const std::string& model_weights_path) {
  auto promise = std::make_shared<folly::Promise<bool>>();
  auto future = promise->getSemiFuture();
  threadpool_.schedule([this, model_weights_path, promise]() mutable {
    bool status = this->init_model(model_weights_path);
    promise->setValue(status);
  });
  return future;
}

std::optional<DiTForwardOutput> DiTWorker::step(const DiTForwardInput& inputs) {
  Timer timer;

  auto output = dit_model_executor_->forward(inputs.to(device_, dtype_));

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  return output;
}

folly::SemiFuture<std::optional<DiTForwardOutput>> DiTWorker::step_async(
    const DiTForwardInput& inputs) {
  auto promise =
      std::make_shared<folly::Promise<std::optional<DiTForwardOutput>>>();
  auto future = promise->getSemiFuture();
  threadpool_.schedule([this, inputs, promise]() mutable {
    auto output = this->step(inputs);
    promise->setValue(output);
  });
  return future;
}

void DiTWorker::process_group_test() {
  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  parallel_state::reduce(tensor, context_.get_parallel_args().process_group_);
  // call allgather
  parallel_state::gather(tensor, context_.get_parallel_args().process_group_);
}

folly::SemiFuture<folly::Unit> DiTWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test();
    promise.setValue();
  });
  return future;
}

// prepare input for execution
DiTForwardInput DiTWorker::prepare_inputs(DiTBatch& batch) {
  return dit_model_executor_->prepare_inputs(batch);
}

int64_t DiTWorker::get_active_activation_memory() {
  return DeviceMonitor::get_instance()
      .get_device_stats(device_.index())
      .active_activation_memory;
}

}  // namespace xllm
