#pragma once

#include <optional>
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

  void add_kv_blocks(const std::vector<Block>& new_blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks,
                            size_t current_total_num_tokens);

  size_t current_max_tokens_capacity() const;

  // returns allocated cache blocks
  Slice<Block> kv_blocks() const;
  std::vector<Block>* mutable_kv_blocks();
  // get the number of blocks
  size_t num_kv_blocks() const;
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end);

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  void reset();

 private:
  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  // kv cache blocks.
  std::vector<Block> blocks_;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // shared blocks number of the sequence.
  uint32_t num_owned_shared_blocks_ = 0;
};

}  // namespace xllm
