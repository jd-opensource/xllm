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
#include "npu/npu_qwen3_moe_decoder_layer_impl.h"
#else
#include "common/qwen3_moe_decoder_layer.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class Qwen3MoeDecoderLayer
    : public torch::nn::ModuleHolder<NpuQwen3MoeDecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuQwen3MoeDecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen3MoeDecoderLayerImpl;

  Qwen3MoeDecoderLayer(const ModelContext& context, int32_t layer_id)
      : Qwen3MoeDecoderLayer(
            std::make_shared<NpuQwen3MoeDecoderLayerImpl>(context, layer_id)) {}
};
#else
class Qwen3MoeDecoderLayer
    : public torch::nn::ModuleHolder<Qwen3MoeDecoderImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3MoeDecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3MoeDecoderImpl;

  Qwen3MoeDecoderLayer(const ModelContext& context, int32_t layer_id)
      : Qwen3MoeDecoderLayer(
            std::make_shared<Qwen3MoeDecoderImpl>(context, layer_id)) {}
};

#endif

}  // namespace layer
}  // namespace xllm
