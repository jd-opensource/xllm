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

#include "musa_attention.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "MTTOplib/Attention.h"
#include "MTTOplib/Ops.h"
#include "MTTOplib/WeightReorder.h"

namespace xllm {
namespace layer {
MusaAttentionImpl::MusaAttentionImpl(ModelArgs const& args,
                                     QuantArgs const& quant_args,
                                     ParallelArgs const& parallel_args,
                                     torch::TensorOptions const& options)
    : MUSALayerBaseImpl(options),
      num_heads_(args.n_heads()),
      num_kv_heads_(args.n_kv_heads().value_or(args.n_heads())),
      head_dim_(args.head_dim()),
      q_size_(num_heads_ * head_dim_),
      kv_size_(num_kv_heads_ * head_dim_),
      rms_eps(args.rms_norm_eps()),
      scaling_(std::sqrt(1.0f / head_dim_)),
      hidden_size_(args.hidden_size()) {
  weights_.resize(weight_num_);
}

torch::Tensor MusaAttentionImpl::forward(torch::Tensor& input,
                                         ForwardParams& fwd_params) {
  auto&& cache = fwd_params.kv_cache;
  auto&& in_param = fwd_params.input_params;

  return xllm_musa::QWen3Attn(input,
                              cache.get_k_cache(),
                              cache.get_v_cache(),
                              in_param.block_tables,
                              fwd_params.attn_meta.mrope_cos,
                              fwd_params.positions,
                              weights_,
                              rms_eps,
                              fwd_params.attn_meta.amd);
}

void MusaAttentionImpl::load_state_dict(StateDict const& state_dict) {
  using WeightMeta = std::pair<std::string, std::vector<int64_t>>;
  static int32_t all_loaded = 0;
  std::vector<WeightMeta> meta = {{"q_proj.", {q_size_, hidden_size_}},
                                  {"k_proj.", {kv_size_, hidden_size_}},
                                  {"v_proj.", {kv_size_, hidden_size_}},
                                  {"o_proj.", {hidden_size_, hidden_size_}},
                                  {"q_norm.", {128}},
                                  {"k_norm.", {128}}};

  for (int32_t i = 0; i < meta.size(); ++i) {
    all_loaded += load_weight_common(
        state_dict.get_dict_with_prefix("self_attn." + meta[i].first),
        meta[i].second,
        i);
  }
  all_loaded += load_weight_common(
      state_dict.get_dict_with_prefix("input_layernorm."), {hidden_size_}, 6);

  if (all_loaded == weight_num_) {
    all_loaded = 0;
    weights_ = xllm_musa::ReorderAttn(weights_);
  }
}
}  // namespace layer
}  // namespace xllm
