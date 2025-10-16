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

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/rms_norm.h"
#include "linear.h"
#include "rotary_embedding.h"

namespace xllm {
namespace layer {

class Qwen3AttentionImpl : public torch::nn::Module {
 public:
  Qwen3AttentionImpl() = default;
  Qwen3AttentionImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_dim_;
  int64_t q_size_;
  int64_t kv_size_;
  float scaling_;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RmsNorm q_norm_{nullptr};
  RmsNorm k_norm_{nullptr};
  Attention attn_{nullptr};
  RotaryEmbedding rotary_emb_{nullptr};
};
TORCH_MODULE(Qwen3Attention);

}  // namespace layer
}  // namespace xllm