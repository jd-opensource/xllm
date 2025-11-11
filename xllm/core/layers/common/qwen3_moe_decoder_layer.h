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

#include <torch/torch.h>

#include <functional>

#include "attention.h"
#include "dense_mlp.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "fused_moe.h"
#include "layers/rms_norm.h"
#include "qwen3_attention.h"

namespace xllm {
namespace layer {

class Qwen3MoeDecoderImpl : public torch::nn::Module {
 public:
  explicit Qwen3MoeDecoderImpl(const ModelContext& context, int32_t layer_id);

  ~Qwen3MoeDecoderImpl() {};

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  Qwen3Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};

  RmsNorm input_norm_{nullptr};
  RmsNorm post_norm_{nullptr};
};

}  // namespace layer
}  // namespace xllm
