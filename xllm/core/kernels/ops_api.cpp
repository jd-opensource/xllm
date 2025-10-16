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

namespace {
#if defined(USE_CUDA)
bool support_pdl() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.major >= 9;
}
#endif

}  // namespace

namespace xllm {
namespace kernel {

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
#elif defined(USE_CUDA)
  cuda::apply_rope_pos_ids_cos_sin_cache(params.q,
                                         params.k,
                                         params.q,
                                         params.k,
                                         params.cos_sin,
                                         params.position_ids,
                                         params.interleaved);
#else
  throw std::runtime_error("apply_rotary not implemented");
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
  cuda::act_and_mul(
      params.output, params.input, params.act_mode, support_pdl());
#else
  throw std::runtime_error("active not implemented");
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
#elif defined(USE_CUDA)
  cuda::reshape_paged_cache(params.slot_mapping,
                            params.key,
                            params.value,
                            params.k_cache,
                            params.v_cache);
#else
  throw std::runtime_error("reshape_paged_cache not implemented");
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
                      params.output,
                      params.output_lse,
                      params.enable_cuda_graph,
                      support_pdl());
#else
  throw std::runtime_error("batch_prefill not implemented");
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
#elif defined(USE_CUDA)
  cuda::batch_decode(params.float_workspace_buffer,
                     params.int_workspace_buffer,
                     params.page_locked_int_workspace_buffer,
                     params.query,
                     params.k_cache,
                     params.v_cache,
                     params.q_cu_seq_lens,
                     params.paged_kv_indptr,
                     params.paged_kv_indices,
                     params.paged_kv_last_page_len,
                     params.window_size_left,
                     params.output,
                     params.output_lse,
                     params.enable_cuda_graph,
                     support_pdl());
#else
  throw std::runtime_error("batch_decode not implemented");
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
  cuda::rmsnorm(
      params.output, params.input, params.weight, params.eps, support_pdl());
#else
  throw std::runtime_error("fused_layernorm not implemented");
#endif
}

torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_MLU)
  return mlu::matmul(
      params.a, params.b, params.bias, params.c, params.alpha, params.beta);
#elif defined(USE_CUDA)
  return cuda::matmul(params.a, params.b, params.bias);
#else
  throw std::runtime_error("matmul not implemented");
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
                        params.start_expert_id,
                        params.block_n,
                        params.avg_moe,
                        params.class_reduce_weight,
                        params.class_expert_id,
                        params.w1_quant_flag,
                        params.w2_quant_flag,
                        params.world_size,
                        params.shared_expert_num,
                        params.parallel_mode);
#elif defined(USE_CUDA)
  throw std::runtime_error("fused_moe for cudanot implemented");
#else
  throw std::runtime_error("fused_moe not implemented");
#endif
}
}  // namespace kernel
}  // namespace xllm