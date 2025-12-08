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

#include "config.h"

namespace xllm {
namespace layer {

class Qwen3MoeDecoderLayer
    : public torch::nn::ModuleHolder<Qwen3MoeDecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3MoeDecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3MoeDecoderLayerImpl;

  Qwen3MoeDecoderLayer(const ModelContext& context, int32_t layer_id)
      : Qwen3MoeDecoderLayer(
            std::make_shared<Qwen3MoeDecoderLayerImpl>(context, layer_id)) {}
};

}  // namespace layer
}  // namespace xllm
