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

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "pytorch/adapter/utils/utils.h"
#endif
#include <memory>
#include <optional>
#include <utility>

#include "common/device_memory.h"
#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/dit_model_loader.h"
#include "framework/dit_cache/dit_cache.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "platform/stream_helper.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
DiTWorker::DiTWorker(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const runtime::Options& options)
    : device_(device), options_(options), parallel_args_(parallel_args) {}

bool DiTWorker::init_model(const std::string& model_weights_path) {
  CHECK(dit_model_ == nullptr) << "Model is already initialized.";
  int currentDevId = device_.index();
#if defined(USE_NPU)
  int ret = aclrtSetDevice(currentDevId);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id:" << currentDevId
               << " failed, ret:" << ret;
  }
#elif defined(USE_MLU)
// TODO(mlu): implement mlu set device
#endif

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

  DiTCacheConfig cache_config_;

  // TODO: Optimize ditcache configuration initialization.
  cache_config_.selected_policy = PolicyType::TaylorSeer;
  cache_config_.taylorseer.n_derivatives = 3;
  cache_config_.taylorseer.skip_interval_steps = 3;
  cache_config_.taylorseer.num_inference_steps = 25;
  cache_config_.taylorseer.warmup_steps = 0;

  bool success = DiTCache::get_instance().init(cache_config_);
  CHECK(success) << "DiTCache init failed";

  return true;
}

std::optional<DiTForwardOutput> DiTWorker::step(const DiTForwardInput& inputs) {
#if defined(USE_NPU)
  c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
// TODO(mlu): implement mlu set device
#endif
  Timer timer;

  auto output = dit_model_executor_->forward(inputs.to(device_, dtype_));

  synchronize_stream(device_.index());
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  return output;
}

folly::SemiFuture<folly::Unit> DiTWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test_async();
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
