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
#include "npu/npu_qwen2_vision_encoder_layer_impl.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class Qwen2VisionEncoderLayer
    : public torch::nn::ModuleHolder<NpuQwen2VisionEncoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuQwen2VisionEncoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen2VisionEncoderLayerImpl;

  Qwen2VisionEncoderLayer(const ModelContext& context)
      : ModuleHolder(
            std::make_shared<NpuQwen2VisionEncoderLayerImpl>(context)) {}
};
#endif

}  // namespace layer
}  // namespace xllm