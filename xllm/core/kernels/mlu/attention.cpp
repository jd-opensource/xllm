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
                         const std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         const std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction) {
  tmo::torch_api::reshape_paged_cache(
      key, value, k_cache, v_cache, slot_mapping, direction);
}

void reshape_from_cache(torch::Tensor& key,
                        const std::optional<torch::Tensor>& value,
                        const torch::Tensor& key_cache,
                        const std::optional<torch::Tensor>& value_cache,
                        const torch::Tensor& context_lengths,
                        const int64_t max_context_len,
                        const std::optional<torch::Tensor>& context_seq_offset,
                        const std::optional<torch::Tensor>& block_tables,
                        const std::optional<torch::Tensor>& cache_seq_offset) {
  tmo::torch_api::reshape_from_cache(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     context_lengths,
                                     max_context_len,
                                     context_seq_offset,
                                     block_tables,
                                     cache_seq_offset);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_table,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse) {
  tmo::torch_api::flash_attention(query,
                                  key,
                                  value,
                                  output,
                                  output_lse,
                                  q_cu_seq_lens,
                                  kv_cu_seq_lens,
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
                  int64_t kv_cache_quant_bit_size) {
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

void masked_indexer_select_paged_kv(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& weights,
    const torch::Tensor& kv_cache_block_table,
    const std::optional<torch::Tensor>& cu_seq_q_lens,
    const std::optional<torch::Tensor>& cu_seq_k_lens,
    const std::optional<torch::Tensor>& k_context_lens,
    const std::optional<torch::Tensor>& k_cache_block_table,
    const bool is_prefill,
    const int64_t index_topk,
    const int64_t kv_cache_block_size,
    const double softmax_scale,
    const std::optional<torch::Tensor>& q_scale,
    const std::optional<torch::Tensor>& k_scale_cache,
    const torch::Tensor& sparse_block_table,
    const torch::Tensor& sparse_context_lens) {
  // add one redundant dimension for future extension
  torch::Tensor weights_extended = weights.unsqueeze(-1);
  tmo::torch_api::masked_indexer_select_paged_kv(query,
                                                 k_cache,
                                                 weights_extended,
                                                 kv_cache_block_table,
                                                 cu_seq_q_lens,
                                                 cu_seq_k_lens,
                                                 k_context_lens,
                                                 k_cache_block_table,
                                                 is_prefill,
                                                 index_topk,
                                                 kv_cache_block_size,
                                                 softmax_scale,
                                                 q_scale,
                                                 k_scale_cache,
                                                 sparse_block_table,
                                                 sparse_context_lens);
}

void masked_indexer_select_paged_kv_torch(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& weights,
    const torch::Tensor& kv_cache_block_table,
    const std::optional<torch::Tensor>& cu_seq_q_lens,
    const std::optional<torch::Tensor>& cu_seq_k_lens,
    const std::optional<torch::Tensor>& k_context_lens,
    const std::optional<torch::Tensor>& k_cache_block_table,
    const bool is_prefill,
    const int64_t index_topk,
    const int64_t kv_cache_block_size,
    const double softmax_scale,
    const std::optional<torch::Tensor>& q_scale,
    const std::optional<torch::Tensor>& k_scale_cache,
    torch::Tensor& sparse_block_table,
    torch::Tensor& sparse_context_lens) {
  // This implementation only supports non-quantized dtypes (bfloat16/half)
  // Quantized dtypes should use the fused kernel version
  TORCH_CHECK(
      !q_scale.has_value() || q_scale.value().numel() == 0,
      "masked_indexer_select_paged_kv_torch does not support quantized query");
  TORCH_CHECK(!k_scale_cache.has_value() || k_scale_cache.value().numel() == 0,
              "masked_indexer_select_paged_kv_torch does not support quantized "
              "k_cache");

  auto device = query.device();
  auto base_dtype = torch::kBFloat16;  // Use bfloat16 for computation
  int64_t head_num, head_size, batch_num;

  // Get dimensions based on mode
  if (is_prefill) {
    // Prefill mode: query is [total_seq_q, head_num, head_size]
    head_num = query.size(1);
    head_size = query.size(2);
    batch_num = cu_seq_q_lens.value().size(0) - 1;
  } else {
    // Decode mode: query is [batch_num, seq_q, head_num, head_size]
    batch_num = query.size(0);
    head_num = query.size(2);
    head_size = query.size(3);
  }

  // Convert query to base dtype for computation
  auto query_base = query.to(base_dtype);
  auto k_blks = is_prefill ? 1 : k_cache.size(2);

  // Process each batch
  for (int64_t i = 0; i < batch_num; ++i) {
    int64_t context_lens_i;
    int64_t q_offset, seq_q;
    torch::Tensor ki, qi, weights_i;

    if (is_prefill) {
      // Prefill mode: get context length from cumulative sequence lengths
      context_lens_i = (cu_seq_k_lens.value()[i + 1] - cu_seq_k_lens.value()[i])
                           .item<int64_t>();
      q_offset = cu_seq_q_lens.value()[i].item<int64_t>();
      seq_q = (cu_seq_q_lens.value()[i + 1] - cu_seq_q_lens.value()[i])
                  .item<int64_t>();

      // Get k_cache slice for this batch
      int64_t k_start = cu_seq_k_lens.value()[i].item<int64_t>();
      ki = k_cache.slice(
          0, k_start, k_start + context_lens_i);  // [context_lens_i, head_size]
      ki = ki.to(base_dtype);
      // MQA: repeat for head_num heads
      ki = ki.unsqueeze(0).repeat(
          {head_num, 1, 1});  // [head_num, seq_k, head_size]

      // Get query slice and permute: [seq_q, head_num, head_size] -> [head_num,
      // head_size, seq_q]
      qi = query_base.slice(0, q_offset, q_offset + seq_q).permute({1, 2, 0});
      weights_i = weights.slice(0, q_offset, q_offset + seq_q);
    } else {
      // Decode mode: get context length from k_context_lens
      context_lens_i = k_context_lens.value()[i].item<int64_t>();
      seq_q = query.size(1);

      // Get k_cache from block table
      int64_t blkn_k_i =
          (context_lens_i + k_blks - 1) / k_blks;  // ceil division
      auto block_indices = k_cache_block_table.value()[i].slice(0, 0, blkn_k_i);
      ki = k_cache.index_select(0,
                                block_indices);  // [blkn, 1, k_blks, head_size]
      ki = ki.reshape({-1, head_size});          // [blkn * k_blks, head_size]
      ki = ki.slice(0, 0, context_lens_i);       // [context_lens_i, head_size]
      ki = ki.to(base_dtype);
      // MQA: repeat for head_num heads
      ki = ki.unsqueeze(0).repeat(
          {head_num, 1, 1});  // [head_num, seq_k, head_size]

      // Get query and permute: [seq_q, head_num, head_size] -> [head_num,
      // head_size, seq_q]
      qi = query_base[i].permute({1, 2, 0});
      weights_i = weights[i];
    }

    // Skip if no context
    if (context_lens_i <= 0) {
      continue;
    }

    // Compute logit = ki @ qi: [head_num, seq_k, seq_q]
    auto logit_i = torch::bmm(ki, qi);
    logit_i = torch::relu(logit_i);

    // Permute: [head_num, seq_k, seq_q] -> [seq_k, head_num, seq_q]
    logit_i = logit_i.permute({1, 0, 2});

    // Apply weights and softmax_scale
    // weights shape: [seq_q, head_num] for decode, [seq_q, head_num] for
    // prefill
    auto q_s = weights_i.to(base_dtype) * softmax_scale;

    // Reshape q_s for broadcasting: [seq_q, head_num] -> [1, head_num, seq_q]
    auto q_s_reshaped = q_s.unsqueeze(0).permute({0, 2, 1});

    // Apply scaling: [seq_k, head_num, seq_q] * [1, head_num, seq_q]
    logit_i = logit_i * q_s_reshaped;

    // Sum over heads: [seq_q, seq_k]
    logit_i = logit_i.sum(1).permute({1, 0});

    // Process each query position
    for (int64_t j = 0; j < seq_q; ++j) {
      // Calculate the valid context length for this query position
      int64_t seq_k_masked = context_lens_i - seq_q + j + 1;
      seq_k_masked = std::max(seq_k_masked, static_cast<int64_t>(0));
      int64_t seq_k_distant = std::min(index_topk, seq_k_masked);

      torch::Tensor block_table;
      if (seq_k_distant > 0) {
        // Get the relevant portion of logit
        auto logit_ij = logit_i[j].slice(0, 0, seq_k_masked);

        // Get top-k indices
        auto topk_result = torch::topk(logit_ij, seq_k_distant);
        auto indices = std::get<1>(topk_result);  // [seq_k_distant]

        // Convert to block table indices
        if (kv_cache_block_size == 1) {
          block_table = kv_cache_block_table[i].index_select(0, indices);
        } else {
          // Convert token indices to block indices (use floor division to keep
          // int64)
          auto block_indices =
              torch::div(indices, kv_cache_block_size, "floor");
          auto offset_indices = indices % kv_cache_block_size;
          auto base_blocks =
              kv_cache_block_table[i].index_select(0, block_indices);
          block_table = base_blocks * kv_cache_block_size + offset_indices;
        }
      } else {
        // Empty case: create empty tensor
        block_table = torch::empty(
            {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      }

      // Write results to output tensors
      if (is_prefill) {
        int64_t out_offset = q_offset + j;
        sparse_context_lens[out_offset] = seq_k_distant;
        if (seq_k_distant > 0) {
          sparse_block_table[out_offset].slice(0, 0, seq_k_distant) =
              block_table;
        }
      } else {
        int64_t out_offset = i * seq_q + j;
        sparse_context_lens[out_offset] = seq_k_distant;
        if (seq_k_distant > 0) {
          sparse_block_table[i][j].slice(0, 0, seq_k_distant) = block_table;
        }
      }
    }
  }
}

}  // namespace xllm::kernel::mlu