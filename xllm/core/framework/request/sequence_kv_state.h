/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/util/slice.h"
#include "framework/block/block.h"

namespace xllm {

class KVCacheState {
 public:
  // get the number of tokens in the kvcache
  size_t kv_cache_tokens_num() const;
  void set_kv_cache_tokens_num(size_t num);
  void incr_kv_cache_tokens_num(size_t num);
  // get the number of shared blocks.
  size_t shared_kv_blocks_num() const;
  size_t shared_kv_tokens_num() const;

  void add_kv_blocks(const std::vector<Block>& new_blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks,
                            size_t current_total_num_tokens);
  void incr_shared_kv_blocks_num(size_t num);
  void set_slice_window_size(uint32_t size);
  void update_slice_window_pos();

  size_t current_max_tokens_capacity() const;

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

  // composite block managers: blocks per BlockType (for CompositeBlockManager).
  // Returns the blocks held under `type`, or an empty slice when absent.
  Slice<Block> blocks_of(BlockType type) const;
  // Mutable block list for `type`, created on demand. Used by the block manager
  // when filling per-type blocks.
  std::vector<Block>* mutable_blocks_of(BlockType type);

  // Groups exported to the worker's multi_block_tables, in the fixed
  // kMultiBlockExportOrder. Each pair is (type, &blocks); only types currently
  // present in the map are returned. Empty for the flat KV path.
  std::vector<std::pair<BlockType, const std::vector<Block>*>>
  multi_block_export_view() const;
  // True when this sequence holds any multi_block_tables-exported group
  // (SWA / C4 / C128); equivalent to the old composite_blocks() being
  // non-empty.
  bool has_multi_block_export() const;

  // Per-sequence linear-state / embedding resource block (BlockType::Single).
  bool has_single_block() const;
  int32_t single_block_id() const;
  void set_single_block(Block&& block);
  Block reset_single_block();

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  uint32_t pushed_local_block_count() const {
    return pushed_local_block_count_;
  }
  void set_pushed_local_block_count(uint32_t n) {
    pushed_local_block_count_ = n;
  }

  size_t next_transfer_block_idx() const;
  void set_next_transfer_block_idx(size_t idx);
  void advance_transfer_block_idx(size_t idx);

  void reset();

  void process_beam_search(std::optional<Block> new_block = std::nullopt);

 private:
  // The flat attention KV block list (BlockType::KV), backing the legacy flat
  // read/write views. kv_view() returns a shared static empty list when absent;
  // mutable_kv_view() creates the entry on demand.
  const std::vector<Block>& kv_view() const;
  std::vector<Block>& mutable_kv_view();

  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  // KV cache blocks keyed by cache role. The legacy flat attention KV lives
  // under BlockType::KV (the flat read views below project onto it); DSV4 keeps
  // its SWA / C4 / C128 groups here; the per-sequence linear/embedding resource
  // block lives under BlockType::Single. std::map keeps deterministic iteration
  // for reset / dealloc / debugging, but worker export order is governed by
  // kMultiBlockExportOrder, not by map order.
  std::map<BlockType, std::vector<Block>> composite_blocks_;

  // source kv cache blocks for swap
  std::vector<Block> src_blocks_;

  // if need to swap last block
  bool need_swap_ = false;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // next logical prompt block index that needs PD PUSH transfer.
  size_t next_transfer_block_idx_ = 0;

  // shared blocks number of the sequence (accounted on the KV group).
  uint32_t num_owned_shared_blocks_ = 0;

  // Sliding-window cursor for legacy callers. CompositeBlockManager keeps DSA
  // SWA block vectors in absolute logical block order and leaves expired
  // logical positions invalid.
  uint32_t slice_window_pos_ = 0;
  uint32_t slice_window_size_ = 0;
  uint32_t slice_window_buffer_ = 0;

  // Number of local KV blocks already pushed to the decode instance.
  // Used for incremental push in chunked prefill + PD disagg mode.
  uint32_t pushed_local_block_count_ = 0;
};

}  // namespace xllm
