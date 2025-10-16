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

#include "common/flashinfer_workspace.h"
#include "kernels/ops_api.h"

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

  // for flashinfer
  attn_metadata.paged_kv_indptr = params.paged_kv_indptr;
  attn_metadata.paged_kv_indices = params.paged_kv_indices;
  attn_metadata.paged_kv_last_page_len = params.paged_kv_last_page_len;
  attn_metadata.q_cu_seq_lens = params.q_seq_lens;
  attn_metadata.kv_cu_seq_lens = params.kv_seq_lens;  // cumulative kv seqlens

  bool is_start_loc_match = (params.q_seq_lens_vec == params.kv_seq_lens_vec);
  attn_metadata.is_chunked_prefill = is_prefill && !is_start_loc_match;
  attn_metadata.is_prefill = is_prefill && !attn_metadata.is_chunked_prefill;
  if (!attn_metadata.is_prefill) {
    attn_metadata.block_table = params.block_tables;
    attn_metadata.kv_seq_lens = torch::diff(params.kv_seq_lens);  // kv seqlens
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
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key;
  reshape_paged_cache_params.value = value;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  xllm::kernel::AttentionParams attention_params;
  attention_params.query = query;
  attention_params.output = output;
  attention_params.output_lse = output_lse;
  attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  attention_params.compute_dtype = attn_metadata.compute_dtype;
  // for flashinfer
  attention_params.float_workspace_buffer =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer();
  attention_params.int_workspace_buffer =
      FlashinferWorkspace::get_instance().get_int_workspace_buffer();
  attention_params.page_locked_int_workspace_buffer =
      FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();
  attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;

  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    attention_params.query_start_loc = attn_metadata.query_start_loc;
    attention_params.seq_start_loc = attn_metadata.seq_start_loc;
    attention_params.max_query_len = attn_metadata.max_query_len;

    xllm::kernel::batch_prefill(attention_params);
  } else if (attn_metadata.is_chunked_prefill) {
    attention_params.key = k_cache;
    attention_params.value = v_cache;
    attention_params.query_start_loc = attn_metadata.query_start_loc;
    attention_params.seq_start_loc = attn_metadata.seq_start_loc;
    attention_params.max_query_len = attn_metadata.max_query_len;
    attention_params.block_table = attn_metadata.block_table;

    xllm::kernel::batch_prefill(attention_params);
  } else {
    query = query.view({-1, 1, num_heads_, head_size_});
    output = output.view({-1, 1, num_heads_, head_size_});

    attention_params.query = query;
    attention_params.output = output;
    attention_params.k_cache = k_cache;
    attention_params.v_cache = v_cache;

    // for mlu
    attention_params.block_table = attn_metadata.block_table;
    attention_params.kv_seq_lens = attn_metadata.kv_seq_lens;

    // for flashinfer
    attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
    attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
    attention_params.paged_kv_last_page_len =
        attn_metadata.paged_kv_last_page_len;

    xllm::kernel::batch_decode(attention_params);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
