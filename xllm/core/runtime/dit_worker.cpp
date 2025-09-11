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
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {
DiTWorker::DiTWorker(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const runtime::Options& options)
    : device_(device), options_(options), parallel_args_(parallel_args) {}

bool DiTWorker::init_model(const std::string& model_weights_path) {
  LOG(INFO) << "Initialize DiT model on device: " << device_;
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
  // initialize model
  // ModelArgs tmp_model_args = model_args;
  // context_.set_model_args(tmp_model_args);
  // context_.set_quant_args(quant_args);
  auto loader = std::make_unique<DiTModelLoader>(model_weights_path);
  ModelArgs model_args = loader->model_args();
  model_args.model_type() = "flux";
  torch::TensorOptions options = torch::TensorOptions().device(device_);
  context_ = ModelContext(parallel_args_, model_args, QuantArgs(), options);
  dit_model_ = create_dit_model(context_);
  LOG(INFO) << "DiT Model created.";
  CHECK(dit_model_ != nullptr) << "Failed to create model.";
  dit_model_->load_model(std::move(loader));
  LOG(INFO) << "DiT Model loaded.";
  DiTModelLoader exec_loader(model_weights_path);
  dit_model_executor_ = std::make_unique<DiTExecutor>(
      dit_model_.get(), std::move(exec_loader), options_);

  LOG(INFO) << "DiT Model executor created.";
  return true;
}

std::optional<DiTForwardOutput> DiTWorker::step(const DiTForwardInput& inputs) {
#if defined(USE_NPU)
  c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
// TODO(mlu): implement mlu set device
#endif
  Timer timer;
  // all tensors should be on the same device as model
  auto input_params = inputs.input_params.to(device_, dtype_);
  auto generation_params = inputs.generation_params;

  // call model executor forward to get hidden states
  auto hidden_states =
      dit_model_executor_->forward(input_params, generation_params);

#if defined(USE_NPU)
  torch::npu::synchronize();
#elif defined(USE_MLU)
// TODO(mlu): implement mlu synchronize stream
#endif
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  DiTForwardOutput output;
  output.image = std::move(hidden_states);
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
