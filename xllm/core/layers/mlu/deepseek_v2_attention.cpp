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

#include "deepseek_v2_attention.h"

#include <glog/logging.h>

#include <tuple>

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

DeepseekV2AttentionImpl::DeepseekV2AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : q_lora_rank_(args.q_lora_rank()),
      kv_lora_rank_(args.kv_lora_rank()),
      qk_nope_head_dim_(args.qk_nope_head_dim()),
      qk_rope_head_dim_(args.qk_rope_head_dim()),
      enable_lighting_indexer_(args.index_n_heads() > 0),
      index_topk_(args.index_topk()),
      v_head_dim_(args.v_head_dim()),
      use_fused_mla_qkv_(false) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();
  int64_t max_position_embeddings = args.max_position_embeddings();

  qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  num_local_heads_ = num_heads / tp_size;
  float scaling = std::pow(qk_head_dim_, -0.5f);

  is_per_token_smoothquant_ = quant_args.quant_method() == "smoothquant";

  if (q_lora_rank_ > 0) {
    q_a_proj_ = register_module(
        "q_a_proj",
        ReplicatedLinear(
            hidden_size, q_lora_rank_, false, QuantArgs(), options));
    q_a_layernorm_ = register_module(
        "q_a_layernorm", RMSNorm(q_lora_rank_, args.rms_norm_eps(), options));
    q_b_proj_ = register_module("q_b_proj",
                                ColumnParallelLinear(q_lora_rank_,
                                                     num_heads * qk_head_dim_,
                                                     false,
                                                     false,
                                                     quant_args,
                                                     parallel_args.tp_group_,
                                                     options));
  } else {
    q_proj_ = register_module("q_proj",
                              ColumnParallelLinear(hidden_size,
                                                   num_heads * qk_head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  }

  kv_a_proj_with_mqa_ =
      register_module("kv_a_proj_with_mqa",
                      ReplicatedLinear(hidden_size,
                                       kv_lora_rank_ + qk_rope_head_dim_,
                                       false,
                                       QuantArgs(),
                                       options));
  kv_a_layernorm_ = register_module(
      "kv_a_layernorm", RMSNorm(kv_lora_rank_, args.rms_norm_eps(), options));
  kv_b_proj_ = register_module(
      "kv_b_proj",
      ColumnParallelLinear(kv_lora_rank_,
                           num_heads * (qk_nope_head_dim_ + v_head_dim_),
                           false,
                           false,
                           QuantArgs(),
                           parallel_args.tp_group_,
                           options));

  auto kv_b_proj_weight = kv_b_proj_->weight();
  auto weights =
      kv_b_proj_weight.unflatten(0, {-1, qk_nope_head_dim_ + v_head_dim_});
  w_kc_ = weights.slice(1, 0, qk_nope_head_dim_);
  w_vc_ = weights.slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

  rotary_emb_ =
      register_module("rotary_emb",
                      DeepseekScalingRotaryEmbedding(
                          qk_rope_head_dim_,
                          qk_rope_head_dim_,
                          max_position_embeddings,
                          args.rope_scaling_original_max_position_embeddings(),
                          args.rope_theta(),
                          /*interleaved*/ true,
                          args.rope_scaling_factor(),
                          args.rope_extrapolation_factor(),
                          args.rope_scaling_attn_factor(),
                          args.rope_scaling_beta_fast(),
                          args.rope_scaling_beta_slow(),
                          args.rope_scaling_mscale(),
                          args.rope_scaling_mscale_all_dim(),
                          options));

  if (args.rope_scaling_rope_type() == "deepseek_yarn") {
    float mscale = layer::rotary::yarn_get_mscale(
        args.rope_scaling_factor(), args.rope_scaling_mscale_all_dim());
    scaling *= mscale * mscale;
  }

  if (enable_lighting_indexer_) {
    indexer_ = register_module("indexer",
                               Indexer(hidden_size,
                                       args.index_n_heads(),
                                       args.index_head_dim(),
                                       qk_rope_head_dim_,
                                       args.index_topk(),
                                       q_lora_rank_,
                                       rotary_emb_,
                                       quant_args,
                                       parallel_args,
                                       options));
  }

  attn_ = register_module("attn",
                          Attention(num_local_heads_,
                                    kv_lora_rank_ + qk_rope_head_dim_,
                                    /*num_local_heads=*/1,
                                    kv_lora_rank_,
                                    args.sliding_window(),
                                    scaling,
                                    use_fused_mla_qkv_,
                                    enable_lighting_indexer_));

  o_proj_ = register_module("o_proj",
                            RowParallelLinear(num_heads * v_head_dim_,
                                              hidden_size,
                                              false,
                                              true,
                                              /*reduce=*/false,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));
}

torch::Tensor DeepseekV2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  int64_t q_len = hidden_states.size(0);
  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  torch::Tensor q, k, v, qr;
  torch::Tensor q_input =
      torch::empty({q_len, num_local_heads_, kv_lora_rank_ + qk_rope_head_dim_},
                   hidden_states.options());

  // get q, qr
  if (q_lora_rank_ > 0) {
    auto q_a = q_a_proj_(hidden_states);
    q_a = std::get<0>(q_a_layernorm_(q_a));
    qr = q_a;
    q = q_b_proj_(q_a).view({-1, num_local_heads_, qk_head_dim_});
  } else {
    q = q_proj_(hidden_states).view({-1, num_local_heads_, qk_head_dim_});
  }

  // get q_nope, q_pe
  auto q_vec = q.split({qk_nope_head_dim_, qk_rope_head_dim_}, -1);
  auto q_nope = q_vec[0];
  auto q_pe = q_vec[1];
  // bmm(q_nope, w_kc_)
  auto q_nope_transposed = q_nope.transpose(0, 1);
  auto q_input_slice = q_input.slice(-1, 0, kv_lora_rank_).transpose(0, 1);
  torch::bmm_out(q_input_slice, q_nope_transposed, w_kc_);

  // get k_nope, k_pe
  auto latent_cache = kv_a_proj_with_mqa_(hidden_states);
  auto v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  auto k_input = latent_cache;
  auto k_input_slice = k_input.slice(-1, 0, kv_lora_rank_);
  // pass the output address so that the output can be written to the address
  // directly
  k_input_slice = std::get<0>(kv_a_layernorm_(v_input,
                                              /*residual=*/std::nullopt,
                                              k_input_slice));
  k_input = k_input.unsqueeze(1);
  auto k_pe = k_input.slice(-1, kv_lora_rank_);

  // rope(q_pe, k_pe)
  rotary_emb_(q_pe,
              k_pe,
              positions,
              attn_metadata.q_cu_seq_lens,
              attn_metadata.max_query_len,
              attn_metadata.is_prefill);
  q_input.slice(-1, kv_lora_rank_) = q_pe;

  // reshape q,k,v
  q_input = q_input.view({q_input.size(0), -1});
  k_input = k_input.view({k_input.size(0), -1});
  v_input = v_input.view({v_input.size(0), -1});

  // reshape_paged_cache before attn
  // since the reshape_paged_cache and indexer_ does not involve any
  // communication, we will skip them if it is dummy run in data parallel
  AttentionMetadata attn_indexer_metadata = attn_metadata;
  if (!attn_metadata.is_dummy) {
    if (only_prefill) {
      auto key = k_input.unsqueeze(1);
      torch::Tensor k_cache = kv_cache.get_k_cache();
      xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
      reshape_paged_cache_params.key = key;
      reshape_paged_cache_params.k_cache = k_cache;
      reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
      xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
    }

    // indexer and update index params for attn
    attn_indexer_metadata = attn_metadata;
    attn_indexer_metadata.compute_dtype = "half";
    if (enable_lighting_indexer_) {
      auto index_cache = kv_cache.get_index_cache();
      auto [new_block_tables, new_context_lens] = indexer_(hidden_states,
                                                           qr,
                                                           positions,
                                                           index_cache,
                                                           attn_metadata,
                                                           only_prefill,
                                                           std::nullopt);
      attn_indexer_metadata.block_table = new_block_tables;
      attn_indexer_metadata.kv_seq_lens = new_context_lens;
      attn_indexer_metadata.max_seq_len = index_topk_;
    }
  }

  // mla forward
  auto [attn_output, output_lse] =
      attn_(attn_indexer_metadata, q_input, k_input, v_input, kv_cache);

  // bmm(attn_out, w_vc_)
  attn_output = attn_output.view({-1, num_local_heads_, kv_lora_rank_});
  auto attn_bmm_output = torch::empty({q_len, num_local_heads_, v_head_dim_},
                                      attn_output.options());
  auto attn_bmm_trans_out = attn_bmm_output.transpose(0, 1);
  torch::bmm_out(attn_bmm_trans_out, attn_output.transpose(0, 1), w_vc_);
  attn_output = attn_bmm_output.flatten(1, 2);

  // o_proj
  auto output = o_proj_(attn_output);
  return output;
}

void DeepseekV2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  // load q proj weights
  if (q_proj_) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
  } else {
    q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_a_proj."));
    q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_b_proj."));
    q_a_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("q_a_layernorm."));
  }

  // load kv proj weights
  kv_a_layernorm_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_layernorm."));
  kv_a_proj_with_mqa_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_proj_with_mqa."));
  kv_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("kv_b_proj."));

  // load indexer weights
  if (enable_lighting_indexer_) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }

  // load o proj weights
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));

  // transpose before forward
  if (kv_b_proj_->is_weight_loaded() && !has_trans_) {
    if (use_fused_mla_qkv_) {
      weight_c_ = w_kc_.transpose(1, 2).contiguous();
    }
    w_vc_ = w_vc_.transpose(1, 2);
    has_trans_ = true;
  }
}

}  // namespace layer
}  // namespace xllm
