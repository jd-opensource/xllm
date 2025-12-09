
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

#include "ilu_ops_api.h"
#include "ixinfer.h"
#include "utils.h"

using namespace ixformer;

namespace xllm::kernel::ilu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& key_cache,
                         std::optional<torch::Tensor>& value_cache,
                         torch::Tensor& slot_mapping) {
  auto value_ = value.value_or(torch::Tensor());
  auto value_cache_ = value_cache.value_or(torch::Tensor());

  int64_t key_token_stride = key.stride(0);
  int64_t value_token_stride = 0;
  if (value_.defined()) {
    value_token_stride = value_.stride(0);
  }
  slot_mapping = slot_mapping.to(at::kLong);
  // translate kvcache shape from [n_blocks, block_size, n_heads, head_dim] to
  // (num_blocks, num_heads, block_size, head_size)
  key_cache = key_cache.permute({0, 2, 1, 3}).contiguous();
  if (value_cache_.defined()) {
    value_cache_ = value_cache_.permute({0, 2, 1, 3}).contiguous();
  }
  infer::vllm_reshape_and_cache(key,
                                value_,
                                key_cache,
                                value_cache_,
                                slot_mapping,
                                key_token_stride,
                                value_token_stride);
}

void batch_prefill(torch::Tensor& query,
                   torch::Tensor& key,
                   torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse) {
  double softcap = 0.0;
  bool sqrt_alibi = false;
  auto q_cu_seq_lens_ = q_cu_seq_lens.value_or(torch::Tensor());
  auto kv_cu_seq_lens_ = kv_cu_seq_lens.value_or(torch::Tensor());
  auto q_quant_scale_ = q_quant_scale.value_or(torch::Tensor());
  auto k_quant_scale_ = k_quant_scale.value_or(torch::Tensor());
  auto v_quant_scale_ = v_quant_scale.value_or(torch::Tensor());

  infer::ixinfer_flash_attn_unpad(query,
                                  key,
                                  value,
                                  output,
                                  q_cu_seq_lens_,
                                  kv_cu_seq_lens_,
                                  max_query_len,
                                  max_seq_len,
                                  is_causal,
                                  window_size_left,
                                  window_size_right,
                                  static_cast<double>(scale),
                                  softcap,
                                  sqrt_alibi,
                                  alibi_slope,
                                  c10::nullopt,
                                  output_lse);
}

void batch_decode(torch::Tensor& query,
                  torch::Tensor& k_cache,
                  torch::Tensor& output,
                  torch::Tensor& block_table,
                  torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  bool is_causal,
                  int64_t kv_cache_quant_bit_size) {
  if (query.dim() == 4) {
    query =
        query
            .view({query.size(0) * query.size(1), query.size(2), query.size(3)})
            .contiguous();
  }
  if (output.dim() == 4) {
    output = output
                 .view({output.size(0) * output.size(1),
                        output.size(2),
                        output.size(3)})
                 .contiguous();
  }
  auto v_cache_ = v_cache.value_or(torch::Tensor());
  k_cache = k_cache.permute({0, 2, 1, 3}).contiguous();
  v_cache_ = v_cache_.permute({0, 2, 1, 3}).contiguous();
  int64_t num_kv_heads = k_cache.size(1);
  int64_t page_block_size = k_cache.size(2);
  double softcap = 0.0;
  bool enable_cuda_graph = false;
  bool use_sqrt_alibi = false;
  // check_tensor_contiguous(k_cache, query.dtype());

  infer::vllm_paged_attention(output,
                              query,
                              k_cache,
                              v_cache_,
                              static_cast<int64_t>(num_kv_heads),
                              scale,
                              block_table,
                              seq_lens,
                              page_block_size,
                              max_seq_len,
                              alibi_slope,
                              is_causal,
                              window_size_left,
                              window_size_right,
                              softcap,
                              enable_cuda_graph,
                              use_sqrt_alibi,
                              c10::nullopt);
}

}  // namespace xllm::kernel::ilu