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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "framework/block/block_group.h"
#include "framework/block/block_manager.h"

namespace xllm {

class KVCacheState;

struct BlockAllocatorCapabilities {
  bool supports_prefix_cache = false;
  bool supports_cache_upload = false;
  bool supports_beam_search = false;
  bool supports_raw_block_alloc = false;
};

struct BlockGroupUsage {
  int32_t group_id = 0;
  int64_t free_blocks = 0;
  int64_t used_blocks = 0;
  int64_t total_blocks = 0;
  int64_t needed_blocks = 0;
  int64_t releasable_blocks = 0;
};

struct SequenceAllocEstimate {
  bool can_allocate = false;
  std::vector<BlockGroupUsage> groups;
  int64_t bottleneck_free_sequences = 0;
};

struct BlockAllocatorStats {
  std::vector<BlockGroupUsage> groups;
  int64_t bottleneck_free_sequences = 0;
  double bottleneck_utilization = 0.0;
};

class SequenceBlockAllocator {
 public:
  virtual ~SequenceBlockAllocator() = default;

  virtual bool allocate_sequence(Sequence* sequence,
                                 size_t target_num_tokens) = 0;
  virtual void deallocate_sequence(Sequence* sequence) = 0;

  virtual void allocate_shared_sequence(Sequence* sequence);
  virtual void cache_sequence(Sequence* sequence);

  virtual SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const = 0;
  virtual std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const = 0;

  virtual BlockAllocatorCapabilities capabilities() const = 0;
  virtual BlockAllocatorStats stats() const = 0;
};

class SingleGroupBlockAllocator final : public SequenceBlockAllocator {
 public:
  explicit SingleGroupBlockAllocator(BlockManager* manager);

  bool allocate_sequence(Sequence* sequence, size_t target_num_tokens) override;
  void deallocate_sequence(Sequence* sequence) override;
  void allocate_shared_sequence(Sequence* sequence) override;
  void cache_sequence(Sequence* sequence) override;

  SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const override;
  std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const override;

  BlockAllocatorCapabilities capabilities() const override;
  BlockAllocatorStats stats() const override;

 private:
  BlockManager* manager_ = nullptr;  // not owned
};

class CompositeSequenceBlockAllocator final : public SequenceBlockAllocator {
 public:
  explicit CompositeSequenceBlockAllocator(const CompositeBlockPlan& plan);

  bool allocate_sequence(Sequence* sequence, size_t target_num_tokens) override;
  void deallocate_sequence(Sequence* sequence) override;

  SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const override;
  std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const override;

  BlockAllocatorCapabilities capabilities() const override;
  BlockAllocatorStats stats() const override;

 private:
  std::vector<int64_t> new_block_counts(
      const std::vector<std::vector<Block>>& composite_blocks,
      size_t target_num_tokens) const;
  void validate_sequence_state(const KVCacheState& kv_state) const;

 private:
  std::vector<BlockGroupSpec> group_specs_;
  std::vector<std::unique_ptr<BlockManager>> group_managers_;
};

}  // namespace xllm
