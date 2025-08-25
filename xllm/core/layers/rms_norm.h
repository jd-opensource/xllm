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
#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <torch_npu/csrc/libs/init_npu.h>

#include "npu/npu_rms_norm_impl.h"
#endif

#include <functional>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class RmsNorm : public torch::nn::ModuleHolder<NpuRmsNormImpl> {
 public:
  using torch::nn::ModuleHolder<NpuRmsNormImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuRmsNormImpl;

  RmsNorm(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuRmsNormImpl>(context)) {}
};
#endif

}  // namespace layer
}  // namespace xllm
