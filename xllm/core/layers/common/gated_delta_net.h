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

#include "common/linear.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class GatedDeltaNetImpl : public torch::nn::Module {
 public:
  GatedDeltaNetImpl() = default;
  GatedDeltaNetImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t hidden_size_;
  int64_t num_k_heads_;
  int64_t num_v_heads_;
  int64_t head_k_dim_;
  int64_t head_v_dim_;
  int64_t key_dim_;
  int64_t value_dim_;
  int64_t conv_kernel_size_;
  float layer_norm_epsilon_;
  int64_t tp_size_;
  int64_t tp_rank_;

  ColumnParallelLinear in_proj_qkvz_{nullptr};
  ColumnParallelLinear in_proj_ba_{nullptr};
  ColumnParallelLinear conv1d_{nullptr};
  RowParallelLinear out_proj_{nullptr};
  RMSNorm norm_{nullptr};

  torch::Tensor A_log_;
  torch::Tensor dt_bias_;
};

TORCH_MODULE(GatedDeltaNet);

}  // namespace layer
}  // namespace xllm
