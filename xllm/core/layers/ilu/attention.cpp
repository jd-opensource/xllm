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

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(head_size),
      sliding_window_(sliding_window),
      scale_(scale),
      use_fused_mla_qkv_(false),
      enable_lighting_indexer_(false),
      enable_mla_(false) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             int64_t num_kv_heads,
                             int64_t v_head_dim,
                             int64_t sliding_window,
                             float scale,
                             bool use_fused_mla_qkv,
                             bool enable_lighting_indexer)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(v_head_dim),
      sliding_window_(sliding_window),
      use_fused_mla_qkv_(use_fused_mla_qkv),
      scale_(scale),
      enable_lighting_indexer_(enable_lighting_indexer),
      enable_mla_(FLAGS_enable_mla) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output;
  if (enable_mla_) {
    output = torch::empty({query.size(0), num_heads_ * v_head_dim_},
                          query.options());
  } else {
    output = torch::empty_like(query);
  }
  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  int64_t num_kv_heads = (enable_mla_ && !only_prefill) ? 1 : num_kv_heads_;
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> v_cache;
  std::optional<torch::Tensor> v;
  if (!enable_mla_) {
    v = value.view({-1, num_kv_heads, head_size_});
    v_cache = kv_cache.get_v_cache();
  }

  bool skip_process_cache = enable_mla_ && (only_prefill || use_fused_mla_qkv_);
  if (!skip_process_cache) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key.view({-1, num_kv_heads, head_size_});
    reshape_paged_cache_params.value = v;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (enable_lighting_indexer_ || !only_prefill) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  }

  int64_t head_size = enable_mla_ ? v_head_dim_ : head_size_;
  output = output.view({-1, num_heads_ * head_size});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  int64_t head_size_v = enable_mla_ ? v_head_dim_ : head_size_;
  xllm::kernel::AttentionParams attention_params;
  attention_params.query = query.view({-1, num_heads_, head_size_});
  attention_params.output = output.view({-1, num_heads_, head_size_v});
  attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  attention_params.compute_dtype = attn_metadata.compute_dtype;
  attention_params.max_query_len = attn_metadata.max_query_len;

  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";

  attention_params.key = key.view({-1, num_kv_heads_, head_size_});
  attention_params.value = value.view({-1, num_kv_heads_, head_size_v});
  attention_params.block_table = std::nullopt;

  attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  xllm::kernel::batch_prefill(attention_params);
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  int64_t head_size_v = enable_mla_ ? v_head_dim_ : head_size_;
  xllm::kernel::AttentionParams attention_params;
  attention_params.query = query.view({-1, 1, num_heads_, head_size_});
  attention_params.output = output.view({-1, 1, num_heads_, head_size_v});
  attention_params.output_lse = std::nullopt;
  attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  attention_params.compute_dtype = attn_metadata.compute_dtype;
  attention_params.k_cache = k_cache;
  attention_params.v_cache = v_cache;

  // for mlu
  attention_params.block_table = attn_metadata.block_table;
  attention_params.kv_seq_lens = attn_metadata.kv_seq_lens;

  xllm::kernel::batch_decode(attention_params);
}

}  // namespace layer
}  // namespace xllm
