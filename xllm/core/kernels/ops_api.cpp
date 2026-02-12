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

#include "ops_api.h"

#if defined(USE_MLU)
#include "mlu/mlu_ops_api.h"
#elif defined(USE_NPU)
#include "npu/npu_ops_api.h"
#elif defined(USE_CUDA)
#include "cuda/attention_runner.h"
#include "cuda/cuda_ops_api.h"
#elif defined(USE_ILU)
#include "ilu/ilu_ops_api.h"
#elif defined(USE_MUSA)
#include "cuda/cuda_ops_api.h"
#include "musa/musa_ops_api.h"
#endif

#include <numeric>

#include "common/macros.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel {

void apply_rotary(RotaryParams& params) {
#if defined(USE_MLU)
  mlu::apply_rotary(params.q,
                    params.k,
                    params.sin,
                    params.cos,
                    params.position_ids,
                    params.cu_query_lens,
                    params.interleaved,
                    params.discrete,
                    params.dynamic_ntk,
                    params.max_query_len);
#elif defined(USE_NPU)
  npu::apply_rotary(
      params.q, params.k, params.cos_sin, params.position_ids.value());
#elif defined(USE_CUDA) || defined(USE_MUSA)
  bool is_neox = !params.interleaved;
  torch::Tensor pos_ids;
  torch::Tensor cos_sin;

  if (params.position_ids.has_value()) {
    pos_ids = params.position_ids.value().to(torch::kInt64);
    auto cos_sin_vec = params.cos_sin.chunk(4, -1);
    auto cos = cos_sin_vec[0];
    auto sin = cos_sin_vec[2];
    cos_sin = torch::cat({cos, sin}, -1);
  } else if (params.cu_query_lens.has_value()) {
    auto cu = params.cu_query_lens.value().to(torch::kInt64);
    CHECK(cu.numel() >= 2) << "apply_rotary (CUDA): cu_query_lens must have at "
                              "least 2 elements when "
                              "position_ids is not provided.";
    int64_t seq_len = cu[1].item<int64_t>() - cu[0].item<int64_t>();
    CHECK(seq_len > 0)
        << "apply_rotary (CUDA): invalid sequence length inferred from "
           "cu_query_lens when position_ids is not provided.";
    pos_ids = torch::arange(seq_len,
                            torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(params.q.device()))
                  .contiguous();

    // MRoPE: use precomputed cos/sin [seq_len, head_dim]. Slice first
    // head_dim/2 before cat so kernel gets in-head rotation pairs [seq_len,
    // head_dim].
    const bool use_mrope_precomputed =
        params.cos.defined() && params.sin.defined() &&
        params.cos.size(0) == static_cast<int64_t>(seq_len) &&
        params.sin.size(0) == static_cast<int64_t>(seq_len);
    if (use_mrope_precomputed) {
      const int64_t head_dim = params.cos.size(-1);
      const int64_t rot_half = head_dim / 2;
      auto cos_sliced = params.cos.contiguous().slice(-1, 0, rot_half);
      auto sin_sliced = params.sin.contiguous().slice(-1, 0, rot_half);
      cos_sin = torch::cat({cos_sliced, sin_sliced}, -1);
    } else {
      auto cos_sin_vec = params.cos_sin.chunk(4, -1);
      auto cos = cos_sin_vec[0];
      auto sin = cos_sin_vec[2];
      cos_sin = torch::cat({cos, sin}, -1);
    }
  } else {
    CHECK(false)
        << "apply_rotary (CUDA): neither position_ids nor cu_query_lens "
           "provided; cannot infer positions.";
  }

  cuda::rotary_embedding(pos_ids, params.q, params.k, cos_sin, is_neox);
#elif defined(USE_ILU)
  auto cos_sin_vec = params.cos_sin.chunk(4, -1);
  auto cos = cos_sin_vec[0];
  auto sin = cos_sin_vec[2];
  auto cos_sin = torch::cat({cos, sin}, -1);
  torch::Tensor long_position_ids = params.position_ids.value().to(at::kLong);
  ilu::apply_rope_pos_ids_cos_sin_cache(
      params.q, params.k, cos_sin, long_position_ids, params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void active(ActivationParams& params) {
#if defined(USE_MLU)
  mlu::active(params.input,
              params.output,
              params.bias,
              params.cusum_token_count,
              params.act_mode,
              params.is_gated,
              params.start_expert_id,
              params.expert_size);
#elif defined(USE_NPU)
  params.output = npu::active(params.input, params.act_mode);
#elif defined(USE_CUDA) || defined(USE_MUSA)
  cuda::act_and_mul(params.output, params.input, params.act_mode);
#elif defined(USE_ILU)
  ilu::act_and_mul(params.output, params.input, params.act_mode);
#else
  NOT_IMPLEMENTED();
#endif
}

void reshape_paged_cache(ReshapePagedCacheParams& params) {
#if defined(USE_MLU)
  mlu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping,
                           params.direction);
#elif defined(USE_NPU)
  npu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping);
#elif defined(USE_CUDA) || defined(USE_MUSA)
  cuda::reshape_paged_cache(params.slot_mapping,
                            params.key,
                            params.value.value_or(torch::Tensor()),
                            params.k_cache,
                            params.v_cache.value_or(torch::Tensor()));
#elif defined(USE_ILU)
  // auto v_cache = params.v_cache.value_or(torch::Tensor());
  ilu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping);
#else
  NOT_IMPLEMENTED();
#endif
}

void reshape_from_cache(ReshapeFromCacheParams& params) {
#if defined(USE_MLU)
  mlu::reshape_from_cache(params.key,
                          params.value,
                          params.key_cache,
                          params.value_cache,
                          params.context_lengths,
                          params.max_context_len,
                          params.context_seq_offset,
                          params.block_tables,
                          params.cache_seq_offset);
#else
  NOT_IMPLEMENTED();
#endif
}

void batch_prefill(AttentionParams& params) {
#if defined(USE_MLU)
  std::optional<torch::Tensor> block_tables;
  if (params.attn_metadata.is_chunked_prefill) {
    block_tables = params.attn_metadata.block_table;
  }
  mlu::batch_prefill(params.query,
                     params.key,
                     params.value,
                     params.output,
                     params.output_lse,
                     params.attn_metadata.q_cu_seq_lens,
                     params.attn_metadata.kv_cu_seq_lens,
                     params.alibi_slope,
                     params.attn_bias,
                     params.q_quant_scale,
                     params.k_quant_scale,
                     params.v_quant_scale,
                     params.out_quant_scale,
                     block_tables,
                     params.attn_metadata.max_query_len,
                     params.attn_metadata.max_seq_len,
                     params.scale,
                     params.attn_metadata.is_causal,
                     params.window_size_left,
                     params.window_size_right,
                     params.attn_metadata.compute_dtype,
                     params.return_lse);
#elif defined(USE_NPU)
  npu::batch_prefill(params.query,
                     params.key,
                     params.value,
                     params.attn_mask,
                     params.seq_lens,
                     params.scale,
                     params.output);
#elif defined(USE_CUDA)
  cuda::batch_prefill_with_optional_piecewise_capture(
      params.attn_metadata.plan_info->uri,
      params.attn_metadata.plan_info->plan_info,
      params.float_workspace_buffer,
      params.int_workspace_buffer,
      params.page_locked_int_workspace_buffer,
      params.query,
      params.key,
      params.value,
      params.attn_metadata.q_cu_seq_lens,
      params.attn_metadata.kv_cu_seq_lens,
      params.window_size_left,
      params.scale,
      params.output,
      params.output_lse,
      params.attn_metadata.enable_cuda_graph);
#elif defined(USE_MUSA)
  cuda::batch_prefill(params.attn_metadata.plan_info->uri,
                      params.attn_metadata.plan_info->plan_info,
                      params.float_workspace_buffer,
                      params.int_workspace_buffer,
                      params.page_locked_int_workspace_buffer,
                      params.query,
                      params.key,
                      params.value,
                      params.attn_metadata.q_cu_seq_lens,
                      params.attn_metadata.kv_cu_seq_lens,
                      params.window_size_left,
                      params.scale,
                      params.output,
                      params.output_lse,
                      params.attn_metadata.enable_cuda_graph,
                      params.mask);
#elif defined(USE_ILU)
  ilu::batch_prefill(params.query,
                     params.key,
                     params.value,
                     params.output,
                     params.output_lse,
                     params.attn_metadata.q_cu_seq_lens,
                     params.attn_metadata.kv_cu_seq_lens,
                     params.alibi_slope,
                     params.attn_bias,
                     params.q_quant_scale,
                     params.k_quant_scale,
                     params.v_quant_scale,
                     params.attn_metadata.block_table,
                     params.attn_metadata.max_query_len,
                     params.attn_metadata.max_seq_len,
                     params.scale,
                     params.attn_metadata.is_causal,
                     params.window_size_left,
                     params.window_size_right,
                     params.attn_metadata.compute_dtype,
                     params.return_lse);
#else
  NOT_IMPLEMENTED();
#endif
}

void batch_decode(AttentionParams& params) {
#if defined(USE_MLU)
  mlu::batch_decode(params.query,
                    params.k_cache,
                    params.output,
                    params.attn_metadata.block_table,
                    params.attn_metadata.kv_seq_lens,
                    params.v_cache,
                    params.output_lse,
                    params.q_quant_scale,
                    params.k_cache_quant_scale,
                    params.v_cache_quant_scale,
                    params.out_quant_scale,
                    params.alibi_slope,
                    params.mask,
                    params.attn_metadata.compute_dtype,
                    params.attn_metadata.max_seq_len,
                    params.window_size_left,
                    params.window_size_right,
                    params.scale,
                    params.return_lse,
                    params.kv_cache_quant_bit_size);
#elif defined(USE_NPU)
  if (params.attn_metadata.paged_attention_tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    npu::batch_decode_acl_graph(
        params.query,
        params.k_cache,
        params.v_cache.value_or(torch::Tensor()),
        params.scale,
        params.attn_metadata.block_table,
        params.seq_lens,
        params.attn_metadata.paged_attention_tiling_data,
        params.output);
  } else {
    // Standard PagedAttention path
    npu::batch_decode(params.query,
                      params.k_cache,
                      params.v_cache.value_or(torch::Tensor()),
                      params.scale,
                      params.attn_metadata.block_table,
                      params.seq_lens,
                      params.output);
  }
#elif defined(USE_CUDA) || defined(USE_MUSA)
  cuda::batch_decode(params.attn_metadata.plan_info->uri,
                     params.attn_metadata.plan_info->plan_info,
                     params.float_workspace_buffer,
                     params.int_workspace_buffer,
                     params.page_locked_int_workspace_buffer,
                     params.query,
                     params.k_cache,
                     params.v_cache.value_or(torch::Tensor()),
                     params.attn_metadata.paged_kv_indptr,
                     params.attn_metadata.paged_kv_indices,
                     params.attn_metadata.paged_kv_last_page_len,
                     params.window_size_left,
                     params.scale,
                     params.output,
                     params.output_lse,
                     params.attn_metadata.enable_cuda_graph,
                     params.use_tensor_core,
                     params.attn_metadata.kv_seq_lens,
                     params.attn_metadata.qo_indptr.defined()
                         ? std::make_optional(params.attn_metadata.qo_indptr)
                         : std::nullopt);
#elif defined(USE_ILU)
  torch::Tensor block_tables, kv_seq_lens;
  block_tables = params.attn_metadata.block_table;
  kv_seq_lens = params.attn_metadata.kv_seq_lens;
  ilu::batch_decode(params.query,
                    params.k_cache,
                    params.output,
                    block_tables,
                    kv_seq_lens,
                    params.v_cache,
                    params.output_lse,
                    params.q_quant_scale,
                    params.k_cache_quant_scale,
                    params.v_cache_quant_scale,
                    params.out_quant_scale,
                    params.alibi_slope,
                    params.mask,
                    params.attn_metadata.compute_dtype,
                    params.block_aligned_max_seq_len,
                    params.window_size_left,
                    params.window_size_right,
                    params.scale,
                    params.return_lse,
                    params.attn_metadata.is_causal,
                    params.kv_cache_quant_bit_size);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_layernorm(FusedLayerNormParams& params) {
#if defined(USE_MLU)
  mlu::fused_layernorm(params.input,
                       params.output,
                       params.residual,
                       params.weight,
                       params.beta,
                       params.bias,
                       params.quant_scale,
                       params.residual_out,
                       params.smooth_quant_scale,
                       params.normed_out,
                       params.mode,
                       params.eps,
                       params.store_output_before_norm,
                       params.store_output_after_norm,
                       params.dynamic_quant);
#elif defined(USE_MUSA)
  musa::fused_layernorm(params.input,
                        params.output,
                        params.residual,
                        params.weight,
                        params.beta,
                        params.bias,
                        params.quant_scale,
                        params.residual_out,
                        params.smooth_quant_scale,
                        params.normed_out,
                        params.mode,
                        params.eps,
                        params.store_output_before_norm,
                        params.store_output_after_norm,
                        params.dynamic_quant);
#elif defined(USE_NPU)
  if (params.residual.has_value()) {
    std::tie(params.output, std::ignore, params.residual_out) =
        npu::add_rms_norm(
            params.input, params.residual.value(), params.weight, params.eps);
  } else {
    params.output =
        npu::rms_norm(params.input, params.weight, params.eps, params.mode);
  }
#elif defined(USE_CUDA) || defined(USE_MUSA)
  if (params.residual.has_value()) {
    cuda::fused_add_rms_norm(
        params.input, params.residual.value(), params.weight, params.eps);
    params.output = params.input;
    params.residual_out = params.residual;
  } else {
    cuda::rms_norm(params.output, params.input, params.weight, params.eps);
  }
#elif defined(USE_ILU)
  if (params.residual.has_value()) {
    ilu::residual_layer_norm(params.input,
                             params.output,
                             params.residual,
                             params.weight,
                             params.bias,  // residual_bias
                             params.residual_out,
                             params.eps);
  } else {
    ilu::rms_norm(params.output, params.input, params.weight, params.eps);
  }
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_MLU)
  return mlu::matmul(
      params.a, params.b, params.bias, params.c, params.alpha, params.beta);
#elif defined(USE_NPU)
  return npu::matmul(params.a, params.b, params.bias);
#elif defined(USE_CUDA) || defined(USE_MUSA)
  return cuda::matmul(params.a, params.b, params.bias);
#elif defined(USE_ILU)
  return ilu::matmul(params.a, params.b, params.bias);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor group_gemm(GroupGemmParams& params) {
#if defined(USE_MLU)
  return mlu::group_gemm(params.a,
                         params.b,
                         params.token_count,
                         params.output,
                         params.a_scale,
                         params.b_scale,
                         params.quant_flag,
                         params.max_dim,
                         params.trans_a,
                         params.trans_b,
                         params.a_quant_bit);
#elif defined(USE_ILU)
  return ilu::group_gemm(params.a,
                         params.b,
                         params.token_count,
                         params.combine_idx,
                         params.output);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    MoeActiveTopkParams& params) {
#if defined(USE_MLU)
  return mlu::moe_active_topk(params.input,
                              params.topk,
                              params.num_expert_group,
                              params.topk_group,
                              params.normalize,
                              params.mask,
                              params.normed_by,
                              params.scoring_func,
                              params.route_scale,
                              params.e_score_correction_bias);
#elif defined(USE_ILU)
  return ilu::moe_active_topk(params.input,
                              params.topk,
                              params.num_expert_group,
                              params.topk_group,
                              params.normalize,
                              params.mask,
                              params.normed_by,
                              params.scoring_func,
                              params.route_scale,
                              params.e_score_correction_bias);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_gen_idx(MoeGenIdxParams& params) {
#if defined(USE_MLU)
  return mlu::moe_gen_idx(params.expert_id, params.expert_num);
#elif defined(USE_ILU)
  return ilu::moe_gen_idx(params.expert_id, params.expert_num);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_expand_input(MoeExpandInputParams& params) {
#if defined(USE_MLU)
  return mlu::moe_expand_input(params.input,
                               params.gather_index,
                               params.cusum_token_count,
                               params.start_expert_id,
                               params.expert_size);
#elif defined(USE_ILU)
  return ilu::moe_expand_input(
      params.input, params.gather_index, params.combine_idx, params.topk);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_combine_result(MoeCombineResultParams& params) {
#if defined(USE_MLU)
  return mlu::moe_combine_result(params.input,
                                 params.reduce_weight,
                                 params.gather_ids,
                                 params.residual,
                                 params.cusum_token_count,
                                 params.start_expert_id,
                                 params.expert_size,
                                 params.bias);
#elif defined(USE_ILU)
  return ilu::moe_combine_result(params.input, params.reduce_weight);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_all2all_gen_send_layout(
    MoeAll2AllGenSendLayoutParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_send_layout(params.token_count, params.nrank);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    MoeAll2AllGenGatherIndexParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_gather_index(
      params.token_num, params.pad_num, params.return_cusum_token_count);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_all2all_create(MoeAll2AllCreateParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_create(params.dispatch_token_byte,
                                 params.combine_token_byte,
                                 params.max_expert_num,
                                 params.max_token_num,
                                 params.rank,
                                 params.nrank,
                                 params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_init(MoeAll2AllInitParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_init(params.handle, params.all_exchange_info, params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_dispatch(MoeAll2AllDispatchParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_dispatch(params.handle,
                            params.token_byte,
                            params.token_num,
                            params.send_layout,
                            params.send_token_num,
                            params.recv_layout,
                            params.recv_token_num,
                            params.send_token,
                            params.recv_token);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_combine(MoeAll2AllCombineParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_combine(params.handle,
                           params.token_byte,
                           params.token_num,
                           params.send_src_layout,
                           params.send_dst_layout,
                           params.send_token,
                           params.recv_token);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_destroy(MoeAll2AllDestroyParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_destroy(params.handle, params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    ScaledQuantizeParams& params) {
#if defined(USE_MLU)
  return mlu::scaled_quantize(params.x,
                              params.smooth,
                              params.zero,
                              params.token_count,
                              params.gather_index,
                              params.gather_index_start_position,
                              params.output,
                              params.output_scale,
                              params.act_mode,
                              params.active_coef,
                              params.is_gated,
                              params.quant_type);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor scaled_matmul(ScaledMatmulParams& params) {
#if defined(USE_MLU)
  return mlu::scaled_matmul(params.a,
                            params.b,
                            params.a_scale,
                            params.b_scale,
                            params.output_dtype,
                            params.bias,
                            params.c,
                            params.act_mode,
                            params.quant_bit_size,
                            params.alpha,
                            params.beta,
                            params.use_hp_active,
                            params.a_quant_bit_size,
                            params.a_calib,
                            params.b_calib,
                            params.output);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor apply_top_k_top_p(TopKPParams& params) {
#if defined(USE_MLU)
  return mlu::apply_top_k_top_p(
      params.logits, params.temperatures, params.top_k, params.top_p);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor random_sample(RandomSampleParams& params) {
#if defined(USE_MLU)
  return mlu::random_sample(params.logits);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor rejection_sample(RejectionSampleParams& params) {
#if defined(USE_MLU)
  return mlu::rejection_sample(params.draft_token_ids,
                               params.num_draft_tokens,
                               params.cu_num_draft_tokens,
                               params.draft_probs,
                               params.target_probs,
                               params.bonus_token_ids,
                               params.uniform_rand,
                               params.uniform_probs,
                               params.max_spec_len);
#else
  NOT_IMPLEMENTED();
#endif
}

void masked_indexer_select_paged_kv(MaskedIndexerSelectPagedKVParams& params) {
#if defined(USE_MLU)
  mlu::masked_indexer_select_paged_kv(params.query,
                                      params.k_cache,
                                      params.weights,
                                      params.kv_cache_block_table,
                                      params.cu_seq_q_lens,
                                      params.cu_seq_k_lens,
                                      params.k_context_lens,
                                      params.k_cache_block_table,
                                      params.is_prefill,
                                      params.index_topk,
                                      params.kv_cache_block_size,
                                      params.softmax_scale,
                                      params.q_scale,
                                      params.k_scale_cache,
                                      params.sparse_block_table,
                                      params.sparse_context_lens);
#else
  NOT_IMPLEMENTED();
#endif
}

void gather_split(GatherSplitParams& params) {
#if defined(USE_MLU)
  mlu::gather_split(params.input,
                    params.gather_index,
                    params.valid_token_num,
                    params.output_head,
                    params.output_tail);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_mla_q(FusedMlaQParams& params) {
#if defined(USE_MLU)
  mlu::fused_mla_q(params.q,
                   params.output,
                   params.output_scale,
                   params.output_norm,
                   params.gamma,
                   params.smooth_quant_scale,
                   params.weight_b,
                   params.weight_b_scale,
                   params.weight_c,
                   params.sin,
                   params.cos,
                   params.position_id,
                   params.quant_mode,
                   params.eps,
                   params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_mla_kv(FusedMlaKVParams& params) {
#if defined(USE_MLU)
  mlu::fused_mla_kv(params.input_kv,
                    params.sin,
                    params.cos,
                    params.position_id,
                    params.gamma,
                    params.kv_cache,
                    params.kv_cache_scale,
                    params.slot_mapping,
                    params.cache_bs_id,
                    params.cache_seq_offset,
                    params.quant_mode,
                    params.is_paged_cache,
                    params.eps,
                    params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_indexer_q(FusedIndexerQParams& params) {
#if defined(USE_MLU)
  mlu::fused_indexer_q(params.input_q,
                       params.output,
                       params.output_scale,
                       params.w_q,
                       params.w_q_scale,
                       params.hadamard_matrix,
                       params.sin,
                       params.cos,
                       params.position_id,
                       params.quant_mode);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_indexer_k(FusedIndexerKParams& params) {
#if defined(USE_MLU)
  mlu::fused_indexer_k(params.x,
                       params.wk,
                       params.wproj,
                       params.sin_table,
                       params.cos_table,
                       params.position_id,
                       params.slot_mapping,
                       params.head_weights,
                       params.k_cache,
                       params.k_cache_scale,
                       params.hadamard_matrix);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> grouped_matmul(GroupedMatmulParams& params) {
#if defined(USE_NPU)
  return npu::apply_npu_grouped_matmul(
      params.x, params.weight, params.bias, params.scale, params.offset,
      params.antiquant_scale, params.antiquant_offset, params.per_token_scale,
      params.group_list, params.activation_input,
      params.activation_quant_scale, params.activation_quant_offset,
      params.split_item, params.group_type, params.group_list_type,
      params.act_type, params.tuning_config, params.output_dtype);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_topk_softmax(
    MoeGatingTopkSoftmaxParams& params) {
#if defined(USE_NPU)
  return npu::apply_moe_gating_topk_softmax(
    params.x, params.finished, params.k
  );
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_token_unpermute(MoeTokenUnpermuteParams &params) {
#if defined(USE_NPU)
  return npu::apply_npu_moe_token_unpermute(
    params.permuted_tokens, params.sorted_indices,
    params.probes, params.padded_mode, params.restore_shape
  );
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_init_routing_v2(
    MoeInitRoutingV2Params &params) {
#if defined(USE_NPU)
  return npu::apply_npu_moe_init_routing_v2(
    params.x, params.expert_idx, params.scale,
    params.offset, params.active_num, params.expert_capacity,
    params.expert_num, params.drop_pad_mode, params.expert_tokens_num_type,
    params.expert_tokens_num_flag, params.quant_mode, params.active_expert_range,
    params.row_idx_type
  );
#else
  NOT_IMPLEMENTED();
#endif
}

}  // namespace xllm::kernel
