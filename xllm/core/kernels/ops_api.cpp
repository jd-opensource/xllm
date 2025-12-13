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
#include "cuda/cuda_ops_api.h"
#elif defined(USE_ILU)
#include "ilu/ilu_ops_api.h"
#endif
#include <glog/logging.h>

#include <numeric>

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
#elif defined(USE_CUDA)
  bool is_neox = !params.interleaved;

  auto pos_ids = params.position_ids.value().to(torch::kInt64);
  auto cos_sin_vec = params.cos_sin.chunk(4, -1);
  auto cos = cos_sin_vec[0];
  auto sin = cos_sin_vec[2];
  auto cos_sin = torch::cat({cos, sin}, -1);

  cuda::rotary_embedding(pos_ids, params.q, params.k, cos_sin, is_neox);
#elif defined(USE_ILU)
  torch::Tensor long_position_ids = params.position_ids.value().to(at::kLong);
  ilu::apply_rope_pos_ids_cos_sin_cache(params.q,
                                        params.k,
                                        params.cos_sin,
                                        long_position_ids,
                                        params.interleaved);
#else
  LOG(FATAL) << "apply_rotary not implemented";
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
#elif defined(USE_CUDA)
  cuda::act_and_mul(params.output, params.input, params.act_mode);
#elif defined(USE_ILU)
  ilu::act_and_mul(params.output, params.input, params.act_mode);
#else
  LOG(FATAL) << "active not implemented";
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
#elif defined(USE_CUDA)
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
  LOG(FATAL) << "reshape_paged_cache not implemented";
#endif
}

void batch_prefill(AttentionParams& params) {
#if defined(USE_MLU)
  mlu::batch_prefill(params.query,
                     params.key,
                     params.value,
                     params.output,
                     params.output_lse,
                     params.query_start_loc,
                     params.seq_start_loc,
                     params.alibi_slope,
                     params.attn_bias,
                     params.q_quant_scale,
                     params.k_quant_scale,
                     params.v_quant_scale,
                     params.out_quant_scale,
                     params.block_table,
                     params.max_query_len,
                     params.max_seq_len,
                     params.scale,
                     params.is_causal,
                     params.window_size_left,
                     params.window_size_right,
                     params.compute_dtype,
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
  cuda::batch_prefill(params.float_workspace_buffer,
                      params.int_workspace_buffer,
                      params.page_locked_int_workspace_buffer,
                      params.query,
                      params.key,
                      params.value,
                      params.q_cu_seq_lens,
                      params.kv_cu_seq_lens,
                      params.window_size_left,
                      params.scale,
                      params.output,
                      params.output_lse,
                      params.enable_cuda_graph);
#elif defined(USE_ILU)
  ilu::batch_prefill(params.query,
                     params.key,
                     params.value,
                     params.output,
                     params.output_lse,
                     params.q_cu_seq_lens,
                     params.kv_cu_seq_lens,
                     params.alibi_slope,
                     params.attn_bias,
                     params.q_quant_scale,
                     params.k_quant_scale,
                     params.v_quant_scale,
                     params.max_query_len,
                     params.max_seq_len,
                     params.scale,
                     params.is_causal,
                     params.window_size_left,
                     params.window_size_right,
                     params.compute_dtype,
                     params.return_lse);
#else
  LOG(FATAL) << "batch_prefill not implemented";
#endif
}

void batch_decode(AttentionParams& params) {
#if defined(USE_MLU)
  mlu::batch_decode(params.query,
                    params.k_cache,
                    params.output,
                    params.block_table.value(),
                    params.kv_seq_lens,
                    params.v_cache,
                    params.output_lse,
                    params.q_quant_scale,
                    params.k_cache_quant_scale,
                    params.v_cache_quant_scale,
                    params.out_quant_scale,
                    params.alibi_slope,
                    params.mask,
                    params.compute_dtype,
                    params.max_seq_len,
                    params.window_size_left,
                    params.window_size_right,
                    params.scale,
                    params.return_lse,
                    params.kv_cache_quant_bit_size);
#elif defined(USE_NPU)
  npu::batch_decode(params.query,
                    params.k_cache,
                    params.v_cache.value_or(torch::Tensor()),
                    params.scale,
                    params.block_table.value(),
                    params.seq_lens,
                    params.output);
#elif defined(USE_CUDA)
  params.query = params.query.squeeze(1);
  params.output = params.output.squeeze(1);
  cuda::batch_decode(params.float_workspace_buffer,
                     params.int_workspace_buffer,
                     params.page_locked_int_workspace_buffer,
                     params.query,
                     params.k_cache,
                     params.v_cache.value_or(torch::Tensor()),
                     params.paged_kv_indptr,
                     params.paged_kv_indices,
                     params.paged_kv_last_page_len,
                     params.window_size_left,
                     params.scale,
                     params.output,
                     params.output_lse,
                     params.enable_cuda_graph);
#elif defined(USE_ILU)
  ilu::batch_decode(params.query,
                    params.k_cache,
                    params.output,
                    params.block_table.value(),
                    params.kv_seq_lens,
                    params.v_cache,
                    params.output_lse,
                    params.q_quant_scale,
                    params.k_cache_quant_scale,
                    params.v_cache_quant_scale,
                    params.out_quant_scale,
                    params.alibi_slope,
                    params.mask,
                    params.compute_dtype,
                    params.max_seq_len,
                    params.window_size_left,
                    params.window_size_right,
                    params.scale,
                    params.return_lse,
                    params.is_causal,
                    params.kv_cache_quant_bit_size);
#else
  LOG(FATAL) << "batch_decode not implemented";
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
#elif defined(USE_NPU)
  if (params.residual.has_value()) {
    std::tie(params.output, std::ignore, params.residual_out) =
        npu::add_rms_norm(
            params.input, params.residual.value(), params.weight, params.eps);
  } else {
    params.output =
        npu::rms_norm(params.input, params.weight, params.eps, params.mode);
  }
#elif defined(USE_CUDA)
  if (params.residual.has_value()) {
    cuda::fused_add_rms_norm(
        params.input, params.residual.value(), params.weight, params.eps);
    params.output = params.input;
    params.residual_out = params.residual;
  } else {
    cuda::rms_norm(params.output, params.input, params.weight, params.eps);
  }
#elif defined(USE_ILU)
  ilu::residual_layer_norm(params.input,
                           params.output,
                           params.residual,
                           params.weight,
                           params.beta,  // weight_bias
                           params.bias,  // residual_bias
                           params.residual_out,
                           params.eps);
#else
  LOG(FATAL) << "fused_layernorm not implemented";
#endif
}

torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_MLU)
  return mlu::matmul(
      params.a, params.b, params.bias, params.c, params.alpha, params.beta);
#elif defined(USE_NPU)
  return npu::matmul(params.a, params.b, params.bias);
#elif defined(USE_CUDA)
  return cuda::matmul(params.a, params.b, params.bias);
#elif defined(USE_ILU)
  return ilu::matmul(params.a, params.b, params.bias);
#else
  LOG(FATAL) << "matmul not implemented";
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
#else
  LOG(FATAL) << "group_gemm not implemented";
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
#else
  LOG(FATAL) << "moe_active_topk not implemented";
#endif
}

std::vector<torch::Tensor> moe_gen_idx(MoeGenIdxParams& params) {
#if defined(USE_MLU)
  return mlu::moe_gen_idx(params.expert_id, params.expert_num);
#else
  LOG(FATAL) << "moe_gen_idx not implemented";
#endif
}

torch::Tensor moe_expand_input(MoeExpandInputParams& params) {
#if defined(USE_MLU)
  return mlu::moe_expand_input(params.input,
                               params.gather_index,
                               params.cusum_token_count,
                               params.start_expert_id,
                               params.expert_size);
#else
  LOG(FATAL) << "moe_expand_input not implemented";
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
#else
  LOG(FATAL) << "moe_combine_result not implemented";
#endif
}

torch::Tensor moe_all2all_gen_send_layout(
    MoeAll2AllGenSendLayoutParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_send_layout(params.token_count, params.nrank);
#else
  LOG(FATAL) << "moe_all2all_gen_send_layout not implemented";
#endif
}

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    MoeAll2AllGenGatherIndexParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_gather_index(
      params.token_num, params.pad_num, params.return_cusum_token_count);
#else
  LOG(FATAL) << "moe_all2all_gen_gather_index not implemented";
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
  LOG(FATAL) << "moe_all2all_create not implemented";
#endif
}

void moe_all2all_init(MoeAll2AllInitParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_init(params.handle, params.all_exchange_info, params.device);
#else
  LOG(FATAL) << "moe_all2all_init not implemented";
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
  LOG(FATAL) << "moe_all2all_dispatch not implemented";
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
  LOG(FATAL) << "moe_all2all_combine not implemented";
#endif
}

void moe_all2all_destroy(MoeAll2AllDestroyParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_destroy(params.handle, params.device);
#else
  LOG(FATAL) << "moe_all2all_destroy not implemented";
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
  LOG(FATAL) << "scaled_quantize not implemented";
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
  LOG(FATAL) << "scaled_matmul not implemented";
#endif
}

torch::Tensor apply_top_k_top_p(TopKPParams& params) {
#if defined(USE_MLU)
  return mlu::apply_top_k_top_p(
      params.logits, params.temperatures, params.top_k, params.top_p);
#else
  LOG(FATAL) << "apply_top_k_top_p not implemented";
#endif
}

torch::Tensor random_sample(RandomSampleParams& params) {
#if defined(USE_MLU)
  return mlu::random_sample(params.logits);
#else
  LOG(FATAL) << "random_sample not implemented";
#endif
}

void masked_indexer_select_paged_kv(MaskedIndexerSelectPagedKVParams& params) {
#if defined(USE_MLU)
  mlu::masked_indexer_select_paged_kv(params.is_prefill,
                                      params.query,
                                      params.cu_seq_q_lens,
                                      params.cu_seq_k_lens,
                                      params.q_scale,
                                      params.weights,
                                      params.softmax_scale,
                                      params.k_cache,
                                      params.k_context_lens,
                                      params.k_cache_block_table,
                                      params.k_scale_cache,
                                      params.index_topk,
                                      params.kv_cache_block_table,
                                      params.kv_cache_block_size,
                                      params.new_block_table,
                                      params.new_context_lens,
                                      params.quant_block_size);
#else
  LOG(FATAL) << "masked_indexer_select_paged_kv not implemented";
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
  LOG(FATAL) << "gather_split not implemented";
#endif
}

}  // namespace xllm::kernel
