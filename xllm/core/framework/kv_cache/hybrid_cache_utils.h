/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "framework/model/model_args.h"

namespace xllm {

struct HybridLinearLayerAllocation {
  bool allocate_full_kv = true;
  bool allocate_linear_state = false;
};

inline bool should_enable_hybrid_linear_cache(const ModelArgs& model_args,
                                              bool enable_disagg_pd,
                                              bool enable_kvcache_store,
                                              double host_blocks_factor) {
  if (!has_linear_attention_layers(model_args)) {
    return false;
  }
  return !enable_disagg_pd && !enable_kvcache_store &&
         host_blocks_factor <= 1.0;
}

inline bool should_use_compact_linear_block_table(
    const ModelArgs& model_args,
    bool enable_hybrid_linear_cache) {
  return enable_hybrid_linear_cache && has_linear_attention_layers(model_args);
}

inline HybridLinearLayerAllocation get_hybrid_linear_layer_allocation(
    const ModelArgs& model_args,
    int64_t layer_id,
    bool enable_hybrid_linear_cache) {
  const bool enable_linear_attention = has_linear_attention_layers(model_args);
  const bool is_full_attn_layer =
      !enable_linear_attention || is_full_attention_layer(model_args, layer_id);
  HybridLinearLayerAllocation allocation;
  allocation.allocate_full_kv =
      !enable_hybrid_linear_cache || is_full_attn_layer;
  allocation.allocate_linear_state =
      enable_linear_attention &&
      (!enable_hybrid_linear_cache || !is_full_attn_layer);
  return allocation;
}

inline int64_t count_hybrid_full_attention_layers(
    const ModelArgs& model_args,
    bool enable_hybrid_linear_cache) {
  if (!enable_hybrid_linear_cache) {
    return model_args.n_layers();
  }

  int64_t num_full_attention_layers = 0;
  for (int64_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
    if (get_hybrid_linear_layer_allocation(
            model_args, layer_id, enable_hybrid_linear_cache)
            .allocate_full_kv) {
      ++num_full_attention_layers;
    }
  }
  return num_full_attention_layers;
}

inline int64_t count_hybrid_linear_attention_layers(
    const ModelArgs& model_args,
    bool enable_hybrid_linear_cache) {
  if (!has_linear_attention_layers(model_args)) {
    return 0;
  }
  if (!enable_hybrid_linear_cache) {
    return model_args.n_layers();
  }

  int64_t num_linear_attention_layers = 0;
  for (int64_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
    if (get_hybrid_linear_layer_allocation(
            model_args, layer_id, enable_hybrid_linear_cache)
            .allocate_linear_state) {
      ++num_linear_attention_layers;
    }
  }
  return num_linear_attention_layers;
}

inline std::vector<int32_t> build_hybrid_linear_state_block_table(
    const std::vector<int32_t>& block_ids,
    uint32_t prev_tokens,
    uint32_t total_tokens,
    int32_t block_size) {
  assert(!block_ids.empty() && "linear state block table requires blocks");
  assert(block_size > 0 && "linear state block table requires block_size > 0");
  assert(total_tokens > 0 &&
         "linear state block table requires total_tokens > 0");

  const int64_t max_block_index = static_cast<int64_t>(block_ids.size()) - 1;
  const int64_t dst_block_index = std::min<int64_t>(
      max_block_index, static_cast<int64_t>((total_tokens - 1) / block_size));
  int64_t src_block_index = dst_block_index;
  if (prev_tokens > 0) {
    src_block_index = std::min<int64_t>(
        max_block_index, static_cast<int64_t>((prev_tokens - 1) / block_size));
  }

  return {block_ids[src_block_index], block_ids[dst_block_index]};
}

}  // namespace xllm
