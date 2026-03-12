/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <tuple>

#include "deepseek_v2_attention.h"
#include "layers/mlu/deepseek_v32_sp_context.h"
#include "platform/device.h"

namespace {

torch::Tensor project_sp_q_heads(xllm::layer::ColumnParallelLinear& proj,
                                 const torch::Tensor& input,
                                 int64_t num_local_heads,
                                 int64_t tp_world_size,
                                 int64_t qk_head_dim) {
  torch::Tensor projected = proj->forward(input, /*use_full_w=*/true);
  return projected.view({-1, num_local_heads * tp_world_size, qk_head_dim});
}

}  // namespace

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::forward_sp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_sp::DeepseekV32SPContext& sp_ctx,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill) {
  CHECK(enable_lighting_indexer_)
      << "deepseek_v32 sequence parallel requires lighting indexer.";
  CHECK(is_prefill_or_chunked_prefill)
      << "deepseek_v32 sequence parallel only supports prefill batches.";
  auto k_cache_scale = kv_cache.get_k_cache_scale();
  SPQRPreOut qr_pre = sp_qr_pre(hidden_states);

  std::optional<torch::Tensor> new_block_tables = std::nullopt;
  std::optional<torch::Tensor> new_context_lens = std::nullopt;
  v32_sp::PaddedGatherHandle mla_handle;
  torch::Tensor local_q_input;
  torch::Tensor index_cache;
  std::optional<IndexerSPPreOut> index_pre = std::nullopt;
  v32_sp::PaddedGatherHandle index_handle;

  Device device(hidden_states.device());
  if (sp_comm_stream_ == nullptr) {
    sp_comm_stream_ = device.get_stream_from_pool();
  }
  index_cache = kv_cache.get_index_cache();
  index_pre = indexer_->sp_pre(
      hidden_states, qr_pre.qr, positions, sp_ctx.local_attn_metadata, sp_ctx);
  auto compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    c10::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    index_handle = indexer_->sp_comm(index_pre->k_local, sp_ctx);
  }

  auto mla_pre = sp_mla_pre(
      hidden_states, positions, qr_pre, sp_ctx, hidden_states.options());
  local_q_input = mla_pre.q_input;

  torch::Tensor k_global =
      indexer_->sp_wait_k(index_pre->k_local, index_handle, sp_ctx);
  compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    c10::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    mla_handle = sp_mla_comm(mla_pre.k_input, sp_ctx);
  }
  auto index_out = indexer_->sp_post(
      *index_pre, k_global, index_cache, attn_metadata, sp_ctx.sp_meta, sp_ctx);
  new_block_tables = std::get<0>(index_out);
  new_context_lens = std::get<1>(index_out);
  sp_mla_finish_k(mla_pre, mla_handle, sp_ctx);

  AttentionMetadata attn_indexer_metadata =
      build_mla_attention_metadata(positions,
                                   hidden_states,
                                   mla_pre.qr,
                                   mla_pre.k_input,
                                   attn_metadata,
                                   kv_cache,
                                   k_cache_scale,
                                   is_prefill_or_chunked_prefill,
                                   new_block_tables,
                                   new_context_lens);
  attn_indexer_metadata.q_cu_seq_lens =
      sp_ctx.local_attn_metadata.q_cu_seq_lens;
  attn_indexer_metadata.max_query_len =
      sp_ctx.local_attn_metadata.max_query_len;
  CHECK(attn_full_)
      << "deepseek_v32 sequence parallel requires full attention weights.";
  auto [attn_output_local, output_lse] = attn_full_(attn_indexer_metadata,
                                                    local_q_input,
                                                    mla_pre.k_input,
                                                    mla_pre.v_input,
                                                    kv_cache);
  return project_sp_output(attn_output_local);
}

DeepseekV2AttentionImpl::SPQRPreOut DeepseekV2AttentionImpl::sp_qr_pre(
    const torch::Tensor& hidden_states) {
  SPQRPreOut out;
  if (q_lora_rank_ > 0) {
    out.q = q_a_proj_(hidden_states);
    auto q_a = std::get<0>(q_a_layernorm_(out.q));
    out.qr = q_a;
    out.q = project_sp_q_heads(
        q_b_proj_, q_a, num_local_heads_, tp_world_size_, qk_head_dim_);
  } else {
    out.q = project_sp_q_heads(
        q_proj_, hidden_states, num_local_heads_, tp_world_size_, qk_head_dim_);
  }
  return out;
}

DeepseekV2AttentionImpl::SPMLAPreOut DeepseekV2AttentionImpl::sp_mla_pre(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const SPQRPreOut& qr_pre,
    const v32_sp::DeepseekV32SPContext& sp_ctx,
    const torch::TensorOptions& options) {
  const int32_t dim = -1;
  const int64_t active_num_heads = num_local_heads_ * tp_world_size_;
  SPMLAPreOut out;
  torch::Tensor q_input = torch::empty({hidden_states.size(0),
                                        active_num_heads,
                                        kv_lora_rank_ + qk_rope_head_dim_},
                                       options);
  out.qr = qr_pre.qr;
  const torch::Tensor& q = qr_pre.q;
  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);

  auto q_vec = q.split({qk_nope_head_dim_, qk_rope_head_dim_}, dim);
  auto q_nope = q_vec[0];
  auto q_pe = q_vec[1];
  auto q_nope_transposed = q_nope.transpose(0, 1);
  auto q_input_slice = q_input.slice(dim, 0, kv_lora_rank_).transpose(0, 1);
  torch::bmm_out(q_input_slice, q_nope_transposed, w_kc_);
  rotary_emb_->forward(q_pe,
                       positions,
                       sp_ctx.local_attn_metadata.q_cu_seq_lens,
                       sp_ctx.local_attn_metadata.max_query_len,
                       /*use_prompt_rope=*/false);
  q_input.slice(dim, kv_lora_rank_) = q_pe;
  decode_kv_pre_base(latent_cache,
                     positions,
                     sp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);
  out.v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  out.k_input = latent_cache;
  out.q_input = q_input.view({q_input.size(0), -1});
  out.k_input = out.k_input.view({out.k_input.size(0), -1});
  out.v_input = out.v_input.view({out.v_input.size(0), -1});
  return out;
}

v32_sp::PaddedGatherHandle DeepseekV2AttentionImpl::sp_mla_comm(
    const torch::Tensor& k_input,
    const v32_sp::DeepseekV32SPContext& sp_ctx) const {
  auto padded_k = layer::v32_sp::pad_to_sp_rows(k_input, sp_ctx);
  return layer::v32_sp::launch_gather_padded(padded_k, sp_ctx);
}

void DeepseekV2AttentionImpl::sp_mla_finish_k(
    SPMLAPreOut& pre_out,
    const v32_sp::PaddedGatherHandle& k_handle,
    const v32_sp::DeepseekV32SPContext& sp_ctx) const {
  pre_out.k_input = layer::v32_sp::finish_gather_padded(k_handle, sp_ctx);
  pre_out.v_input = pre_out.k_input.slice(-1, 0, kv_lora_rank_);
}

torch::Tensor DeepseekV2AttentionImpl::project_sp_output(
    const torch::Tensor& attn_output) {
  const int64_t active_num_heads = num_local_heads_ * tp_world_size_;
  auto attn_output_view =
      attn_output.view({-1, active_num_heads, kv_lora_rank_});
  auto attn_bmm_output =
      torch::empty({attn_output.size(0), active_num_heads, v_head_dim_},
                   attn_output.options());
  auto attn_bmm_trans_out = attn_bmm_output.transpose(0, 1);
  torch::bmm_out(attn_bmm_trans_out, attn_output_view.transpose(0, 1), w_vc_);
  auto proj_input = attn_bmm_output.flatten(1, 2);
  return o_proj_->forward(proj_input, /*use_full_w=*/true);
}

}  // namespace layer
}  // namespace xllm
