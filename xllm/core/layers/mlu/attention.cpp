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

#include "attention.h"

#include "kernels/mlu/torch_ops_api.h"

namespace xllm::mlu {

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           bool is_prefill) {
  return AttentionMetadata::build(params, "float", is_prefill);
}

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           const std::string& compute_dtype,
                                           bool is_prefill) {
  AttentionMetadata attn_metadata;
  attn_metadata.query_start_loc = params.q_seq_lens;
  attn_metadata.seq_start_loc = params.kv_seq_lens;
  attn_metadata.max_query_len = params.q_max_seq_len;
  attn_metadata.max_seq_len = params.kv_max_seq_len;
  attn_metadata.slot_mapping = params.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;
  attn_metadata.is_prefill = is_prefill;
  if (!is_prefill) {
    attn_metadata.block_table = params.block_tables;
    attn_metadata.seq_lens = torch::diff(params.kv_seq_lens);
  }
  return attn_metadata;
}

Attention::Attention(int num_heads,
                     int head_size,
                     float scale,
                     int num_kv_heads,
                     int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window) {}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Attention::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  auto output = torch::empty_like(query);
  auto output_lse = c10::nullopt;

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  reshape_paged_cache(key,
                      value,
                      k_cache,
                      v_cache,
                      attn_metadata.slot_mapping,
                      false /* direction */);

  if (attn_metadata.is_prefill) {
    flash_attention(query,
                    key,
                    value,
                    output,
                    output_lse,
                    attn_metadata.query_start_loc,
                    attn_metadata.seq_start_loc,
                    c10::nullopt /* alibi_slope */,
                    c10::nullopt /* attn_bias */,
                    c10::nullopt /* q_quant_scale */,
                    c10::nullopt /* k_quant_scale */,
                    c10::nullopt /* v_quant_scale */,
                    c10::nullopt /* out_quant_scale */,
                    c10::nullopt /* block_tables */,
                    attn_metadata.max_query_len,
                    attn_metadata.max_seq_len,
                    scale_,
                    true /* is_causal */,
                    sliding_window_,
                    sliding_window_,
                    attn_metadata.compute_dtype,
                    false /* return_lse */);
  } else {
    query = query.view({-1, 1, num_heads_, head_size_});
    output = output.view({-1, 1, num_heads_, head_size_});
    single_query_cached_kv_attn(query,
                                k_cache,
                                output,
                                attn_metadata.block_table,
                                attn_metadata.seq_lens,
                                v_cache,
                                output_lse,
                                c10::nullopt /* q_quant_scale */,
                                c10::nullopt /* k_cache_quant_scale */,
                                c10::nullopt /* v_cache_quant_scale */,
                                c10::nullopt /* out_quant_scale */,
                                c10::nullopt /* alibi_slope */,
                                attn_metadata.compute_dtype,
                                attn_metadata.max_seq_len,
                                sliding_window_,
                                -1 /* always -1 for window size right */,
                                scale_,
                                false /* return_lse */,
                                -1 /* kv_cache_quant_bit_size */);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace xllm::mlu
