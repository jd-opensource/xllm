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

#include "core/framework/model_context.h"

#include <torch/torch.h>
#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <torch_npu/csrc/libs/init_npu.h>
#endif

namespace xllm {
ModelContext::ModelContext(const ParallelArgs& input_parallel_args,
                           const ModelArgs& model_args,
                           const QuantArgs& quant_args,
                           const torch::TensorOptions& tensor_options)
    : parallel_args_(input_parallel_args),
      model_args_(model_args),
      quant_args_(quant_args),
      tensor_options_(tensor_options) {
#if defined(USE_NPU)
  int32_t device_id = tensor_options.device().index();
  aclError ret = aclrtSetDevice(device_id);
  atb::CreateContext(&context_);
  void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
  context_->SetExecuteStream(stream);
  context_->SetAsyncTilingCopyStatus(true);
  atb_workspace_ = std::make_shared<AtbWorkspace>(tensor_options.device());
#endif
}

#if defined(USE_NPU)
ModelContext::ModelContext(const ParallelArgs& input_parallel_args,
                           const ModelArgs& model_args,
                           const QuantArgs& quant_args,
                           const torch::TensorOptions& tensor_options,
                           atb::Context* context)
    : parallel_args_(input_parallel_args),
      model_args_(model_args),
      quant_args_(quant_args),
      tensor_options_(tensor_options),
      context_(context) {}
#endif

}  // namespace xllm
