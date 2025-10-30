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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void reshape_paged_cache(torch::Tensor& key,
                         torch::Tensor& value,
                         torch::Tensor& k_cache,
                         torch::Tensor& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction) {
  tmo::torch_api::reshape_paged_cache(
      key, value, k_cache, v_cache, slot_mapping, direction);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& query_start_loc,
                   const std::optional<torch::Tensor>& seq_start_loc,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_table,
                   int max_query_len,
                   int max_seq_len,
                   float scale,
                   bool is_causal,
                   int window_size_left,
                   int window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse) {
  tmo::torch_api::flash_attention(query,
                                  key,
                                  value,
                                  output,
                                  output_lse,
                                  query_start_loc,
                                  seq_start_loc,
                                  alibi_slope,
                                  attn_bias,
                                  q_quant_scale,
                                  k_quant_scale,
                                  v_quant_scale,
                                  out_quant_scale,
                                  block_table,
                                  max_query_len,
                                  max_seq_len,
                                  scale,
                                  is_causal,
                                  window_size_left,
                                  window_size_right,
                                  compute_dtype,
                                  return_lse);
}

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const torch::Tensor& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int max_seq_len,
                  int window_size_left,
                  int window_size_right,
                  float scale,
                  bool return_lse,
                  int kv_cache_quant_bit_size) {
  tmo::torch_api::single_query_cached_kv_attn(query,
                                              k_cache,
                                              output,
                                              block_table,
                                              seq_lens,
                                              v_cache,
                                              output_lse,
                                              q_quant_scale,
                                              k_cache_quant_scale,
                                              v_cache_quant_scale,
                                              out_quant_scale,
                                              alibi_slope,
                                              mask,
                                              compute_dtype,
                                              max_seq_len,
                                              window_size_left,
                                              window_size_right,
                                              scale,
                                              return_lse,
                                              kv_cache_quant_bit_size);
}

}  // namespace xllm::kernel::mlu