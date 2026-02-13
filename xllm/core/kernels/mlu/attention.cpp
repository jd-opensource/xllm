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
  // Only supports non-quantized dtypes; use fused kernel for quantized versions
  TORCH_CHECK(
      !q_scale.has_value() || q_scale.value().numel() == 0,
      "masked_indexer_select_paged_kv_torch does not support quantized query");
  TORCH_CHECK(!k_scale_cache.has_value() || k_scale_cache.value().numel() == 0,
              "masked_indexer_select_paged_kv_torch does not support quantized "
              "k_cache");

  auto device = query.device();
  auto base_dtype = torch::kBFloat16;
  int64_t head_num, head_size, batch_num;

  // ========== OPTIMIZED DECODE PATH (Fully Vectorized) ==========
  if (!is_prefill) {
    auto idx_options =
        torch::TensorOptions().device(device).dtype(torch::kInt32);

    int64_t batch_size = query.size(0);
    int64_t seq_q = query.size(1);
    head_num = query.size(2);
    head_size = query.size(3);

    int64_t max_seq_k = k_context_lens.value().max().item<int64_t>();

    if (max_seq_k == 0) {
      sparse_context_lens.zero_();
      return;
    }

    // Build dense K cache tensor: gather all tokens from paged memory
    auto seq_indices = torch::arange(max_seq_k, idx_options);
    auto seq_indices_exp =
        seq_indices.unsqueeze(0).expand({batch_size, max_seq_k});

    auto block_idx = torch::div(seq_indices_exp, kv_cache_block_size, "floor");
    auto block_offset = seq_indices_exp % kv_cache_block_size;

    auto physical_blocks = k_cache_block_table.value().gather(1, block_idx);
    auto physical_tokens = physical_blocks * kv_cache_block_size + block_offset;

    auto k_flat = k_cache.reshape({-1, head_size});
    auto k_dense = k_flat.index_select(0, physical_tokens.flatten());
    k_dense = k_dense.view({batch_size, max_seq_k, head_size});

    // MQA/GQA broadcast
    k_dense = k_dense.unsqueeze(1)
                  .expand({batch_size, head_num, max_seq_k, head_size})
                  .to(base_dtype);

    // Compute attention scores: Q @ K^T
    auto q_dense = query.to(base_dtype).permute({0, 2, 1, 3});
    auto k_transposed = k_dense.transpose(-1, -2);
    auto logit = torch::matmul(q_dense, k_transposed);

    // Apply ReLU, weights, and scale
    logit = torch::relu(logit);
    auto w_scaled = (weights.to(base_dtype) * softmax_scale)
                        .view({batch_size, head_num, 1, 1});
    logit = logit * w_scaled;
    logit = logit.sum(1);

    // Mask out padded tokens (use -1e4 to avoid NaN from -inf)
    auto context_lens_exp = k_context_lens.value().unsqueeze(1);
    auto valid_mask = seq_indices_exp < context_lens_exp;
    valid_mask = valid_mask.unsqueeze(1);
    logit.masked_fill_(~valid_mask, -1e4);

    // Extract top-K indices
    int64_t actual_topk = std::min(index_topk, max_seq_k);
    auto topk_out = torch::topk(logit, actual_topk, /*dim=*/-1);
    auto topk_indices = std::get<1>(topk_out);

    // Calculate valid lengths and map to physical indices
    auto seq_k_masked =
        torch::clamp(k_context_lens.value() - seq_q + 1, /*min=*/0);
    auto seq_k_distant = torch::clamp_max(seq_k_masked, index_topk);

    auto out_block_idx =
        torch::div(topk_indices, kv_cache_block_size, "floor").squeeze(1);
    auto out_block_offset = (topk_indices % kv_cache_block_size).squeeze(1);

    auto base_blocks_out = k_cache_block_table.value().gather(1, out_block_idx);
    auto final_physical_indices =
        base_blocks_out * kv_cache_block_size + out_block_offset;

    // Write results
    sparse_context_lens.slice(0, 0, batch_size).copy_(seq_k_distant);

    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t valid_k = seq_k_distant[i].item<int64_t>();
      if (valid_k > 0) {
        sparse_block_table[i][0].slice(0, 0, valid_k) =
            final_physical_indices[i].slice(0, 0, valid_k).to(torch::kInt32);
      }
    }

    return;
  }

  // ========== PREFILL PATH ==========
  torch::Tensor cu_seq_q_lens_cpu;
  torch::Tensor cu_seq_k_lens_cpu;
  torch::Tensor k_context_lens_cpu;

  if (is_prefill) {
    head_num = query.size(1);
    head_size = query.size(2);
    batch_num = cu_seq_q_lens.value().size(0) - 1;
  } else {
    batch_num = query.size(0);
    head_num = query.size(2);
    head_size = query.size(3);
  }

  // Pre-copy to CPU to avoid per-iteration synchronization
  if (cu_seq_q_lens.has_value()) {
    cu_seq_q_lens_cpu = cu_seq_q_lens.value().to(torch::kCPU);
  }
  if (cu_seq_k_lens.has_value()) {
    cu_seq_k_lens_cpu = cu_seq_k_lens.value().to(torch::kCPU);
  }
  if (k_context_lens.has_value()) {
    k_context_lens_cpu = k_context_lens.value().to(torch::kCPU);
  }

  auto query_base = query.to(base_dtype);
  auto k_blks = is_prefill ? 1 : k_cache.size(2);

  for (int64_t i = 0; i < batch_num; ++i) {
    int64_t context_lens_i;
    int64_t q_offset, seq_q;
    torch::Tensor ki, qi, weights_i;

    auto cu_q_accessor = cu_seq_q_lens_cpu.accessor<int32_t, 1>();
    auto cu_k_accessor = cu_seq_k_lens_cpu.accessor<int32_t, 1>();

    context_lens_i = cu_k_accessor[i + 1] - cu_k_accessor[i];
    q_offset = cu_q_accessor[i];
    seq_q = cu_q_accessor[i + 1] - cu_q_accessor[i];

    int64_t k_start = cu_k_accessor[i];
    ki = k_cache.slice(0, k_start, k_start + context_lens_i);
    ki = ki.to(base_dtype);
    ki = ki.unsqueeze(0).repeat({head_num, 1, 1});

    qi = query_base.slice(0, q_offset, q_offset + seq_q).permute({1, 2, 0});
    weights_i = weights.slice(0, q_offset, q_offset + seq_q);

    if (context_lens_i <= 0) {
      continue;
    }

    // Compute attention scores
    auto logit_i = torch::bmm(ki, qi);
    logit_i = torch::relu(logit_i);
    logit_i = logit_i.permute({1, 0, 2});

    auto q_s = weights_i.to(base_dtype) * softmax_scale;
    auto q_s_reshaped = q_s.unsqueeze(0).permute({0, 2, 1});
    logit_i = logit_i * q_s_reshaped;
    logit_i = logit_i.sum(1).permute({1, 0});

    for (int64_t j = 0; j < seq_q; ++j) {
      int64_t seq_k_masked = context_lens_i - seq_q + j + 1;
      seq_k_masked = std::max(seq_k_masked, static_cast<int64_t>(0));
      int64_t seq_k_distant = std::min(index_topk, seq_k_masked);

      torch::Tensor block_table;
      if (seq_k_distant > 0) {
        auto logit_ij = logit_i[j].slice(0, 0, seq_k_masked);
        auto topk_result = torch::topk(logit_ij, seq_k_distant);
        auto indices = std::get<1>(topk_result);

        if (kv_cache_block_size == 1) {
          block_table = kv_cache_block_table[i].index_select(0, indices);
        } else {
          auto block_indices =
              torch::div(indices, kv_cache_block_size, "floor");
          auto offset_indices = indices % kv_cache_block_size;
          auto base_blocks =
              kv_cache_block_table[i].index_select(0, block_indices);
          block_table = base_blocks * kv_cache_block_size + offset_indices;
        }
      } else {
        block_table = torch::empty(
            {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      }

      int64_t out_offset = q_offset + j;
      sparse_context_lens[out_offset] = seq_k_distant;
      if (seq_k_distant > 0) {
        sparse_block_table[out_offset].slice(0, 0, seq_k_distant) = block_table;
      }
    }
  }
}

}  // namespace xllm::kernel::mlu