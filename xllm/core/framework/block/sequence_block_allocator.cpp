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

#include "framework/block/sequence_block_allocator.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "framework/block/block_manager_impl.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

int64_t ceil_div(size_t value, int32_t divisor) {
  const size_t typed_divisor = static_cast<size_t>(divisor);
  return static_cast<int64_t>((value + typed_divisor - 1) / typed_divisor);
}

int64_t block_count(size_t size) { return static_cast<int64_t>(size); }

int64_t releasable_block_count(const Slice<Block>& blocks,
                               bool enable_prefix_cache) {
  const uint32_t max_ref_count = enable_prefix_cache ? 2 : 1;
  int64_t count = 0;
  for (const Block& block : blocks) {
    if (block.is_valid() && block.ref_count() <= max_ref_count) {
      ++count;
    }
  }
  return count;
}

size_t total_blocks(const std::vector<std::vector<Block>>& blocks) {
  size_t total = 0;
  for (const std::vector<Block>& group_blocks : blocks) {
    total += group_blocks.size();
  }
  return total;
}

void fill_bottleneck(SequenceAllocEstimate& estimate) {
  estimate.can_allocate = true;
  int64_t bottleneck = std::numeric_limits<int64_t>::max();
  for (const BlockGroupUsage& group : estimate.groups) {
    if (group.needed_blocks > group.free_blocks) {
      estimate.can_allocate = false;
    }
    int64_t margin = group.needed_blocks == 0
                         ? group.free_blocks
                         : group.free_blocks / group.needed_blocks;
    bottleneck = std::min(bottleneck, margin);
  }
  estimate.bottleneck_free_sequences =
      bottleneck == std::numeric_limits<int64_t>::max() ? 0 : bottleneck;
}

double group_utilization(const BlockGroupUsage& group) {
  if (group.total_blocks == 0) {
    return 0.0;
  }
  return static_cast<double>(group.used_blocks) /
         static_cast<double>(group.total_blocks);
}

}  // namespace

SingleGroupBlockAllocator::SingleGroupBlockAllocator(BlockManager* manager)
    : manager_(manager) {}

bool SingleGroupBlockAllocator::allocate_sequence(Sequence* sequence,
                                                  size_t target_num_tokens) {
  const size_t current_blocks = sequence->kv_state().num_kv_blocks();
  const size_t block_size = manager_->block_size();
  const size_t needed_blocks =
      (target_num_tokens + block_size - 1) / block_size;
  if (needed_blocks <= current_blocks) {
    return true;
  }

  const size_t extra_blocks = needed_blocks - current_blocks;
  std::vector<Block> blocks = manager_->allocate(extra_blocks);
  if (blocks.size() != extra_blocks) {
    return false;
  }

  sequence->add_kv_blocks(blocks);
  return true;
}

void SingleGroupBlockAllocator::deallocate_sequence(Sequence* sequence) {
  manager_->deallocate(sequence->kv_state().kv_blocks());
}

SequenceAllocEstimate SingleGroupBlockAllocator::estimate_allocate(
    const Sequence* sequence,
    size_t target_num_tokens) const {
  const KVCacheState& state = sequence->kv_state();
  const size_t current_blocks = state.num_kv_blocks();
  const int64_t target_blocks =
      ceil_div(target_num_tokens, static_cast<int32_t>(manager_->block_size()));
  const int64_t needed_blocks = std::max<int64_t>(
      0, target_blocks - static_cast<int64_t>(current_blocks));

  SequenceAllocEstimate estimate;
  BlockGroupUsage group;
  group.group_id = 0;
  group.free_blocks = static_cast<int64_t>(manager_->available_blocks());
  group.used_blocks = static_cast<int64_t>(manager_->num_used_blocks());
  group.total_blocks = static_cast<int64_t>(manager_->num_total_blocks());
  group.needed_blocks = needed_blocks;
  estimate.groups.emplace_back(group);
  fill_bottleneck(estimate);
  return estimate;
}

std::vector<BlockGroupUsage> SingleGroupBlockAllocator::estimate_release(
    const Sequence* sequence) const {
  BlockGroupUsage group;
  group.group_id = 0;
  group.free_blocks = static_cast<int64_t>(manager_->available_blocks());
  group.used_blocks = static_cast<int64_t>(manager_->num_used_blocks());
  group.total_blocks = static_cast<int64_t>(manager_->num_total_blocks());
  group.releasable_blocks =
      releasable_block_count(sequence->kv_state().kv_blocks(),
                             manager_->options().enable_prefix_cache());
  return {group};
}

BlockAllocatorStats SingleGroupBlockAllocator::stats() const {
  BlockAllocatorStats stats;
  BlockGroupUsage group;
  group.group_id = 0;
  group.free_blocks = static_cast<int64_t>(manager_->num_free_blocks());
  group.used_blocks = static_cast<int64_t>(manager_->num_used_blocks());
  group.total_blocks = static_cast<int64_t>(manager_->num_total_blocks());
  stats.groups.emplace_back(group);
  stats.bottleneck_free_sequences = group.free_blocks;
  stats.bottleneck_utilization = group_utilization(group);
  return stats;
}

CompositeSequenceBlockAllocator::CompositeSequenceBlockAllocator(
    const Plan& plan) {
  group_specs_ = plan.groups;
  group_managers_.reserve(group_specs_.size());

  for (const GroupSpec& group : group_specs_) {
    BlockManager::Options options;
    options.num_blocks(static_cast<uint32_t>(group.num_blocks))
        .block_size(group.tokens_per_block)
        .enable_prefix_cache(false)
        .enable_cache_upload(false);
    group_managers_.emplace_back(std::make_unique<BlockManagerImpl>(options));
  }
}

bool CompositeSequenceBlockAllocator::allocate_sequence(
    Sequence* sequence,
    size_t target_num_tokens) {
  std::vector<std::vector<Block>>* composite_blocks =
      sequence->kv_state().mutable_composite_blocks();
  if (composite_blocks->empty()) {
    composite_blocks->resize(group_specs_.size());
    std::vector<KVCacheGroupState>* group_states =
        sequence->kv_state().mutable_composite_group_states();
    group_states->clear();
    group_states->reserve(group_specs_.size());
    for (const GroupSpec& group : group_specs_) {
      KVCacheGroupState group_state;
      group_state.group_id = group.group_id;
      group_state.is_token_group = group.kind == GroupKind::TOKEN;
      group_state.tokens_per_block = group.tokens_per_block;
      group_states->emplace_back(group_state);
    }
  }
  validate_sequence_state(sequence->kv_state());

  const std::vector<int64_t> extra_blocks =
      new_block_counts(*composite_blocks, target_num_tokens);
  for (size_t i = 0; i < extra_blocks.size(); ++i) {
    const int64_t available_blocks =
        static_cast<int64_t>(group_managers_[i]->available_blocks());
    if (extra_blocks[i] > available_blocks) {
      if (total_blocks(*composite_blocks) == 0) {
        sequence->kv_state().clear_composite_blocks();
      }
      return false;
    }
  }

  std::vector<size_t> old_sizes;
  old_sizes.reserve(composite_blocks->size());
  for (const std::vector<Block>& blocks : *composite_blocks) {
    old_sizes.emplace_back(blocks.size());
  }

  const auto rollback = [&]() {
    for (size_t i = 0; i < composite_blocks->size(); ++i) {
      std::vector<Block>& group_blocks = composite_blocks->at(i);
      if (group_blocks.size() <= old_sizes[i]) {
        continue;
      }
      std::vector<Block> blocks(
          group_blocks.begin() + static_cast<int64_t>(old_sizes[i]),
          group_blocks.end());
      group_managers_[i]->deallocate(blocks);
      group_blocks.resize(old_sizes[i]);
    }
    if (total_blocks(*composite_blocks) == 0) {
      sequence->kv_state().clear_composite_blocks();
    }
  };

  for (size_t i = 0; i < extra_blocks.size(); ++i) {
    if (extra_blocks[i] == 0) {
      continue;
    }
    std::vector<Block> blocks =
        group_managers_[i]->allocate(static_cast<size_t>(extra_blocks[i]));
    if (blocks.size() != static_cast<size_t>(extra_blocks[i])) {
      rollback();
      return false;
    }
    std::vector<Block>& group_blocks = composite_blocks->at(i);
    group_blocks.insert(group_blocks.end(), blocks.begin(), blocks.end());
  }
  return true;
}

void CompositeSequenceBlockAllocator::deallocate_sequence(Sequence* sequence) {
  std::vector<std::vector<Block>>* composite_blocks =
      sequence->kv_state().mutable_composite_blocks();
  CHECK_LE(composite_blocks->size(), group_managers_.size());
  for (size_t i = 0; i < composite_blocks->size(); ++i) {
    if (composite_blocks->at(i).empty()) {
      continue;
    }
    group_managers_[i]->deallocate(composite_blocks->at(i));
  }
  sequence->kv_state().clear_composite_blocks();
}

SequenceAllocEstimate CompositeSequenceBlockAllocator::estimate_allocate(
    const Sequence* sequence,
    size_t target_num_tokens) const {
  const std::vector<std::vector<Block>>& composite_blocks =
      sequence->kv_state().composite_blocks();
  if (!composite_blocks.empty()) {
    validate_sequence_state(sequence->kv_state());
  }

  const std::vector<int64_t> extra_blocks =
      new_block_counts(composite_blocks, target_num_tokens);

  SequenceAllocEstimate estimate;
  estimate.groups.reserve(group_specs_.size());
  for (size_t i = 0; i < group_specs_.size(); ++i) {
    BlockGroupUsage group;
    group.group_id = group_specs_[i].group_id;
    group.free_blocks =
        static_cast<int64_t>(group_managers_[i]->available_blocks());
    group.used_blocks =
        static_cast<int64_t>(group_managers_[i]->num_used_blocks());
    group.total_blocks =
        static_cast<int64_t>(group_managers_[i]->num_total_blocks());
    group.needed_blocks = extra_blocks[i];
    estimate.groups.emplace_back(group);
  }
  fill_bottleneck(estimate);
  return estimate;
}

std::vector<BlockGroupUsage> CompositeSequenceBlockAllocator::estimate_release(
    const Sequence* sequence) const {
  const std::vector<std::vector<Block>>& composite_blocks =
      sequence->kv_state().composite_blocks();

  std::vector<BlockGroupUsage> usages;
  usages.reserve(group_specs_.size());
  for (size_t i = 0; i < group_specs_.size(); ++i) {
    BlockGroupUsage group;
    group.group_id = group_specs_[i].group_id;
    group.free_blocks =
        static_cast<int64_t>(group_managers_[i]->available_blocks());
    group.used_blocks =
        static_cast<int64_t>(group_managers_[i]->num_used_blocks());
    group.total_blocks =
        static_cast<int64_t>(group_managers_[i]->num_total_blocks());
    if (i < composite_blocks.size()) {
      group.releasable_blocks = block_count(composite_blocks[i].size());
    }
    usages.emplace_back(group);
  }
  return usages;
}

BlockAllocatorStats CompositeSequenceBlockAllocator::stats() const {
  BlockAllocatorStats stats;
  stats.groups.reserve(group_specs_.size());
  int64_t bottleneck = std::numeric_limits<int64_t>::max();
  double bottleneck_utilization = 0.0;

  for (size_t i = 0; i < group_specs_.size(); ++i) {
    BlockGroupUsage group;
    group.group_id = group_specs_[i].group_id;
    group.free_blocks =
        static_cast<int64_t>(group_managers_[i]->num_free_blocks());
    group.used_blocks =
        static_cast<int64_t>(group_managers_[i]->num_used_blocks());
    group.total_blocks =
        static_cast<int64_t>(group_managers_[i]->num_total_blocks());
    stats.groups.emplace_back(group);
    bottleneck = std::min(bottleneck, group.free_blocks);
    bottleneck_utilization =
        std::max(bottleneck_utilization, group_utilization(group));
  }

  stats.bottleneck_free_sequences =
      bottleneck == std::numeric_limits<int64_t>::max() ? 0 : bottleneck;
  stats.bottleneck_utilization = bottleneck_utilization;
  return stats;
}

std::vector<int64_t> CompositeSequenceBlockAllocator::new_block_counts(
    const std::vector<std::vector<Block>>& composite_blocks,
    size_t target_num_tokens) const {
  std::vector<int64_t> new_counts(group_specs_.size(), 0);
  for (size_t i = 0; i < group_specs_.size(); ++i) {
    const GroupSpec& group = group_specs_[i];
    const int64_t current_blocks = i < composite_blocks.size()
                                       ? block_count(composite_blocks[i].size())
                                       : 0;
    switch (group.kind) {
      case GroupKind::RING:
        if (current_blocks == 0) {
          new_counts[i] = group.fixed_blocks_per_sequence;
        } else {
          CHECK_EQ(current_blocks, group.fixed_blocks_per_sequence)
              << "RING block group must be empty or fully allocated";
        }
        break;
      case GroupKind::TOKEN: {
        const int64_t target_blocks =
            ceil_div(target_num_tokens, group.tokens_per_block);
        new_counts[i] = std::max<int64_t>(0, target_blocks - current_blocks);
        break;
      }
      default:
        LOG(FATAL) << "Unknown block group kind";
    }
  }
  return new_counts;
}

void CompositeSequenceBlockAllocator::validate_sequence_state(
    const KVCacheState& kv_state) const {
  const std::vector<std::vector<Block>>& composite_blocks =
      kv_state.composite_blocks();
  if (composite_blocks.empty()) {
    return;
  }
  CHECK_EQ(composite_blocks.size(), group_specs_.size())
      << "Composite sequence block groups must match the plan";
  const std::vector<KVCacheGroupState>& group_states =
      kv_state.composite_group_states();
  if (group_states.empty()) {
    return;
  }
  CHECK_EQ(group_states.size(), group_specs_.size())
      << "Composite sequence group state must match the plan";
  for (size_t i = 0; i < group_specs_.size(); ++i) {
    CHECK_EQ(group_states[i].group_id, group_specs_[i].group_id);
    CHECK_EQ(group_states[i].is_token_group,
             group_specs_[i].kind == GroupKind::TOKEN);
    CHECK_EQ(group_states[i].tokens_per_block,
             group_specs_[i].tokens_per_block);
  }
}

}  // namespace xllm
