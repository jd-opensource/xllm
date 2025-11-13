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
#include "npu/npu_deepseek_v2_decoder_layer_impl.h"
#else
#include "common/deepseek_v2_decoder_layer.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class DeepseekV2DecoderLayer
    : public torch::nn::ModuleHolder<NpuDeepseekV2DecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuDeepseekV2DecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuDeepseekV2DecoderLayerImpl;

  DeepseekV2DecoderLayer(const ModelContext& context,
                         const int32_t layer_id,
                         const float sm_scale)
      : ModuleHolder(
            std::make_shared<NpuDeepseekV2DecoderLayerImpl>(context,
                                                            layer_id,
                                                            sm_scale)) {}
};
#else
// DeepSeek V3.2 used different structure but
// it is still compatible with DeepSeek V2.
class DeepseekV2DecoderLayer
    : public torch::nn::ModuleHolder<DeepseekV2DecoderImpl> {
 public:
  using torch::nn::ModuleHolder<DeepseekV2DecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = DeepseekV2DecoderImpl;

  DeepseekV2DecoderLayer(const ModelContext& context, const int32_t layer_id)
      : ModuleHolder(
            std::make_shared<DeepseekV2DecoderImpl>(context, layer_id)) {}
};
#endif

}  // namespace layer
}  // namespace xllm
