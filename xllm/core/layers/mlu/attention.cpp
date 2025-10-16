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

#include "kernels/torch_ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

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

  bool is_start_loc_match = (params.q_seq_lens_vec == params.kv_seq_lens_vec);
  attn_metadata.is_chunked_prefill = is_prefill && !is_start_loc_match;
  attn_metadata.is_prefill = is_prefill && !attn_metadata.is_chunked_prefill;
  if (!attn_metadata.is_prefill) {
    attn_metadata.block_table = params.block_tables;
    attn_metadata.seq_lens = torch::diff(params.kv_seq_lens);
  }

  return attn_metadata;
}

AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key;
  reshape_paged_cache_params.value = value;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (!attn_metadata.is_prefill) {
    query = query.view({-1, 1, num_heads_, head_size_});
    output = output.view({-1, 1, num_heads_, head_size_});

    DecodeAttentionParams decode_attention_params;
    decode_attention_params.query = query;
    decode_attention_params.k_cache = k_cache;
    decode_attention_params.output = output;
    decode_attention_params.block_table = attn_metadata.block_table;
    decode_attention_params.seq_lens = attn_metadata.seq_lens;
    decode_attention_params.v_cache = v_cache;
    decode_attention_params.output_lse = output_lse;
    decode_attention_params.compute_dtype = attn_metadata.compute_dtype;
    decode_attention_params.max_seq_len = attn_metadata.max_seq_len;
    decode_attention_params.window_size_left = sliding_window_;
    decode_attention_params.scale = scale_;

    xllm::kernel::decode_attention(decode_attention_params);
  } else {
    PrefillAttentionParams prefill_attention_params;
    prefill_attention_params.query = query;
    prefill_attention_params.key = key;
    prefill_attention_params.value = value;
    prefill_attention_params.output = output;
    prefill_attention_params.output_lse = output_lse;
    prefill_attention_params.query_start_loc = attn_metadata.query_start_loc;
    prefill_attention_params.seq_start_loc = attn_metadata.seq_start_loc;
    prefill_attention_params.max_query_len = attn_metadata.max_query_len;
    prefill_attention_params.max_seq_len = attn_metadata.max_seq_len;
    prefill_attention_params.scale = scale_;
    prefill_attention_params.window_size_left = sliding_window_;
    prefill_attention_params.compute_dtype = attn_metadata.compute_dtype;

    if (!attn_metadata.is_chunked_prefill) {
      prefill_attention_params.key = key;
      prefill_attention_params.value = value;
    } else {
      prefill_attention_params.key = k_cache;
      prefill_attention_params.value = v_cache;
      prefill_attention_params.block_tables = attn_metadata.block_table;
    }

    xllm::kernel::prefill_attention(prefill_attention_params);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
