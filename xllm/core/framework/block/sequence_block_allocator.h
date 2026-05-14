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

#include "framework/block/block_manager.h"

namespace xllm {

class KVCacheState;

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

class SequenceBlockAllocator {
 public:
  virtual ~SequenceBlockAllocator() = default;

  virtual bool allocate_sequence(Sequence* sequence,
                                 size_t target_num_tokens) = 0;
  virtual void deallocate_sequence(Sequence* sequence) = 0;

  virtual SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const = 0;
  virtual std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const = 0;

  virtual int64_t bottleneck_free_blocks() const = 0;
  virtual double bottleneck_utilization() const = 0;
  virtual int64_t max_used_blocks() const = 0;
};

class SingleGroupBlockAllocator final : public SequenceBlockAllocator {
 public:
  explicit SingleGroupBlockAllocator(BlockManager* manager);

  bool allocate_sequence(Sequence* sequence, size_t target_num_tokens) override;
  void deallocate_sequence(Sequence* sequence) override;

  SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const override;
  std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const override;

  int64_t bottleneck_free_blocks() const override;
  double bottleneck_utilization() const override;
  int64_t max_used_blocks() const override;

 private:
  BlockManager* manager_ = nullptr;  // not owned
};

class CompositeSequenceBlockAllocator final : public SequenceBlockAllocator {
 public:
  enum class GroupKind : int32_t {
    TOKEN = 0,
    RING = 1,
  };

  struct GroupSpec {
    int32_t group_id = 0;
    GroupKind kind = GroupKind::TOKEN;
    int32_t block_size = 0;
    int64_t num_blocks = 0;
    int32_t fixed_blocks_per_sequence = 0;
  };

  struct Plan {
    std::vector<GroupSpec> groups;
  };

  explicit CompositeSequenceBlockAllocator(const Plan& plan);

  bool allocate_sequence(Sequence* sequence, size_t target_num_tokens) override;
  void deallocate_sequence(Sequence* sequence) override;

  SequenceAllocEstimate estimate_allocate(
      const Sequence* sequence,
      size_t target_num_tokens) const override;
  std::vector<BlockGroupUsage> estimate_release(
      const Sequence* sequence) const override;

  int64_t bottleneck_free_blocks() const override;
  double bottleneck_utilization() const override;
  int64_t max_used_blocks() const override;

 private:
  std::vector<int64_t> new_block_counts(
      const std::vector<std::vector<Block>>& composite_blocks,
      size_t target_num_tokens) const;
  void validate_sequence_state(const KVCacheState& kv_state) const;

 private:
  std::vector<GroupSpec> group_specs_;
  std::vector<std::unique_ptr<BlockManager>> group_managers_;
};

}  // namespace xllm
