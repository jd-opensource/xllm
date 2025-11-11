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
  npu::apply_rotary(params.q, params.k, params.cos_sin, params.positions);
#elif defined(USE_CUDA)
  cuda::apply_rope_pos_ids_cos_sin_cache(params.q,
                                         params.k,
                                         params.cos_sin,
                                         params.position_ids.value(),
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
#elif defined(USE_CUDA)
  cuda::act_and_mul(params.output, params.input, params.act_mode);
#else
  LOG(FATAL) << "active not implemented";
#endif
}

torch::Tensor active_tensor(ActivationParams& params) {
#if defined(USE_NPU)
  return npu::active(params.input, params.act_mode);
#else
  LOG(FATAL) << "active_tensor not implemented";
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
                      params.paged_kv_indptr,
                      params.window_size_left,
                      params.scale,
                      params.output,
                      params.output_lse,
                      params.enable_cuda_graph);
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
                    params.v_cache,
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
                     params.v_cache,
                     params.paged_kv_indptr,
                     params.paged_kv_indices,
                     params.paged_kv_last_page_len,
                     params.window_size_left,
                     params.scale,
                     params.output,
                     params.output_lse,
                     params.enable_cuda_graph);
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
#elif defined(USE_CUDA)
  cuda::rmsnorm(params.output, params.input, params.weight, params.eps);
#else
  LOG(FATAL) << "fused_layernorm not implemented";
#endif
}

torch::Tensor fused_layernorm_tensor(FusedLayerNormParams& params) {
#if defined(USE_NPU)
  return npu::fused_layernorm(
      params.input, params.weight, params.eps, params.mode);
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
#else
  LOG(FATAL) << "matmul not implemented";
#endif
}

torch::Tensor fused_moe(FusedMoEParams& params) {
#if defined(USE_MLU)
  return mlu::fused_moe(params.hidden_states,
                        params.gating_output,
                        params.w1,
                        params.w2,
                        params.bias1,
                        params.bias2,
                        params.residual,
                        params.input_smooth,
                        params.act_smooth,
                        params.w1_scale,
                        params.w2_scale,
                        params.e_score_correction_bias,
                        params.topk,
                        params.renormalize,
                        params.gated,
                        params.act_mode,
                        params.scoring_func,
                        params.num_expert_group,
                        params.topk_group,
                        params.route_scale,
                        params.start_expert_id,
                        params.avg_moe,
                        params.w1_quant_flag,
                        params.w2_quant_flag);
#elif defined(USE_CUDA)
  LOG(FATAL) << "fused_moe for cuda not implemented";
#else
  LOG(FATAL) << "fused_moe not implemented";
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

}  // namespace xllm::kernel
