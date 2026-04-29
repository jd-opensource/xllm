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

#include "framework/batch/transfer_kv_info_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {

namespace {

uint32_t div_ceil(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

void append_xtensor_offsets(TransferKVInfo* info,
                            const TransferKVInfo& full_info,
                            const std::vector<size_t>& remote_idxs) {
  if (full_info.dst_xtensor_layer_offsets.empty()) {
    return;
  }

  info->dst_xtensor_layer_offsets.reserve(
      full_info.dst_xtensor_layer_offsets.size());
  for (const XTensorLayerOffsets& full_layer :
       full_info.dst_xtensor_layer_offsets) {
    CHECK_EQ(full_layer.k_offsets.size(), full_info.remote_blocks_ids.size());
    CHECK_EQ(full_layer.v_offsets.size(), full_info.remote_blocks_ids.size());
    XTensorLayerOffsets layer;
    layer.k_offsets.reserve(remote_idxs.size());
    layer.v_offsets.reserve(remote_idxs.size());
    for (size_t remote_idx : remote_idxs) {
      CHECK_LT(remote_idx, full_layer.k_offsets.size());
      CHECK_LT(remote_idx, full_layer.v_offsets.size());
      layer.k_offsets.emplace_back(full_layer.k_offsets[remote_idx]);
      layer.v_offsets.emplace_back(full_layer.v_offsets[remote_idx]);
    }
    info->dst_xtensor_layer_offsets.emplace_back(std::move(layer));
  }
}

}  // namespace

TransferKVInfo build_step_transfer_info(
    const TransferKVInfo& full_info,
    const std::vector<uint64_t>& local_block_ids,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t block_size) {
  TransferKVInfo info;
  info.request_id = full_info.request_id;
  info.dp_rank = full_info.dp_rank;
  info.remote_instance_info = full_info.remote_instance_info;

  if (block_size == 0 || local_block_ids.empty() ||
      full_info.remote_blocks_ids.empty()) {
    return info;
  }

  const size_t local_size = local_block_ids.size();
  const size_t remote_size = full_info.remote_blocks_ids.size();
  const size_t full_size = full_info.local_blocks_ids.empty()
                               ? local_size
                               : full_info.local_blocks_ids.size();
  const size_t shared_blocks =
      full_size > remote_size ? full_size - remote_size : 0;
  const size_t win_begin = static_cast<size_t>(n_kv_cache_tokens / block_size);
  const size_t win_end = static_cast<size_t>(div_ceil(seq_len, block_size));
  const size_t map_begin = std::max(win_begin, shared_blocks);
  const size_t map_limit = std::min(local_size, shared_blocks + remote_size);
  const size_t map_end = std::min(win_end, map_limit);

  if (map_begin >= map_end) {
    return info;
  }

  std::vector<size_t> remote_idxs;
  const size_t block_cnt = map_end - map_begin;
  info.local_blocks_ids.reserve(block_cnt);
  info.remote_blocks_ids.reserve(block_cnt);
  remote_idxs.reserve(block_cnt);
  for (size_t local_idx = map_begin; local_idx < map_end; ++local_idx) {
    const size_t remote_idx = local_idx - shared_blocks;
    info.local_blocks_ids.emplace_back(local_block_ids[local_idx]);
    info.remote_blocks_ids.emplace_back(
        full_info.remote_blocks_ids[remote_idx]);
    remote_idxs.emplace_back(remote_idx);
  }

  append_xtensor_offsets(&info, full_info, remote_idxs);
  return info;
}

}  // namespace xllm
