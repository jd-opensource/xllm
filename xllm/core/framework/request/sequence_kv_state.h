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

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/common/types.h"
#include "core/util/slice.h"
#include "framework/block/block.h"

namespace xllm {

struct KVCacheGroupState {
  int32_t group_id = 0;
  bool is_token_group = true;
  int32_t tokens_per_block = 0;
};

class KVCacheState {
 public:
  // get the number of tokens in the kvcache
  size_t kv_cache_tokens_num() const;
  void set_kv_cache_tokens_num(size_t num);
  void incr_kv_cache_tokens_num(size_t num);
  // get the number of shared blocks.
  size_t shared_kv_blocks_num() const;

  void add_kv_blocks(const std::vector<Block>& new_blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks,
                            size_t current_total_num_tokens);
  void incr_shared_kv_blocks_num(size_t num);

  size_t current_max_tokens_capacity() const;
  size_t token_capacity() const;

  // returns allocated cache blocks
  Slice<Block> kv_blocks() const;
  std::vector<Block>* mutable_kv_blocks();

  Slice<Block> src_blocks() const { return src_blocks_; };

  void set_src_blocks(const std::vector<Block>& src_blocks,
                      bool need_swap = false) {
    src_blocks_ = std::move(src_blocks);
    need_swap_ = need_swap;
  };

  bool need_swap() const { return need_swap_; }

  // get the number of blocks
  size_t num_kv_blocks() const;
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end);

  const std::vector<std::vector<Block>>& composite_blocks() const {
    return composite_blocks_;
  }
  std::vector<std::vector<Block>>* mutable_composite_blocks() {
    return &composite_blocks_;
  }
  const std::vector<KVCacheGroupState>& composite_group_states() const {
    return composite_group_states_;
  }
  std::vector<KVCacheGroupState>* mutable_composite_group_states() {
    return &composite_group_states_;
  }
  void clear_composite_blocks();
  bool has_composite_blocks() const;
  size_t num_composite_groups() const;
  size_t num_composite_blocks(int32_t group_id) const;
  size_t total_composite_blocks() const;

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  void reset();

  void process_beam_search(std::optional<Block> new_block = std::nullopt);

 private:
  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  // kv cache blocks.
  std::vector<Block> blocks_;

  // source kv cache blocks for swap
  std::vector<Block> src_blocks_;

  // if need to swap last block
  bool need_swap_ = false;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // shared blocks number of the sequence.
  uint32_t num_owned_shared_blocks_ = 0;

  std::vector<std::vector<Block>> composite_blocks_;
  std::vector<KVCacheGroupState> composite_group_states_;
};

}  // namespace xllm
