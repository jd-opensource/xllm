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

#include "composite_block_manager.h"

#include <algorithm>

#include "block_manager_impl.h"
#include "framework/kv_cache/kv_cache_event.h"
#include "sliding_window_block_manager.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;
// DeepSeek V4/MindIE keeps 12 sliding-window blocks per active sequence, plus
// two reserved rows in the physical cache group.
constexpr uint32_t kSlidingWindowBlocksPerSequence = 12;

}  // namespace

CompositeBlockManager::CompositeBlockManager(
    const BlockManager::Options& options)
    : BlockManager(options) {
  const size_t n = options_.manager_types().size();
  CHECK_EQ(n, options_.compress_ratios().size())
      << "manager_types and compress_ratios must have the same size";
  CHECK_GT(n, 0u) << "CompositeBlockManager requires at least one sub-manager";

  sub_managers_.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const uint32_t type = options_.manager_types()[i];
    const uint32_t compress_ratio = options_.compress_ratios()[i];
    BlockManager::Options opts = options_;

    if (type == kManagerTypeBlockManagerImpl) {
      opts.block_size(static_cast<uint32_t>(options_.block_size()) *
                      compress_ratio);
      opts.num_blocks(static_cast<uint32_t>(options_.num_blocks()) /
                      compress_ratio);
      sub_managers_.push_back(std::make_unique<BlockManagerImpl>(opts));
    } else if (type == kManagerTypeSlidingWindowBlockManager) {
      const uint32_t max_seqs = std::max(options_.max_seqs_per_batch(), 1u);
      const uint32_t swa_total_blocks =
          kSlidingWindowBlocksPerSequence * max_seqs + 2;
      opts.num_blocks(swa_total_blocks)
          .window_size(kSlidingWindowBlocksPerSequence);
      LOG(INFO)
          << "CompositeBlockManager uses sliding-window "
             "allocation: blocks_per_sequence="
          << kSlidingWindowBlocksPerSequence
          << ", total_blocks=" << swa_total_blocks << ", max_seqs=" << max_seqs
          << ". This keeps SW block ids within the physical SW cache rows.";
      sub_managers_.push_back(
          std::make_unique<SlidingWindowBlockManager>(opts));
    } else {
      LOG(FATAL) << "Unknown manager_type " << type;
    }
  }
}

bool CompositeBlockManager::allocate_for_sequence(Sequence* seq,
                                                  size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  std::vector<std::vector<Block>>* composite =
      seq->kv_state().mutable_composite_blocks();
  composite->resize(sub_managers_.size());

  if (composite->at(0).empty()) {
    // slice window manager allocate blocks.
    const size_t swa_blocks_per_sequence =
        sub_managers_[0]->options().window_size();
    composite->at(0) =
        std::move(sub_managers_[0]->allocate(swa_blocks_per_sequence));
    if (composite->at(0).size() != swa_blocks_per_sequence) {
      return false;
    }
    seq->kv_state().set_slice_window_size(sub_managers_[0]->block_size() *
                                          swa_blocks_per_sequence);
  } else {
    seq->kv_state().update_slice_window_pos();
  }
  for (size_t i = 1; i < sub_managers_.size(); ++i) {
    const size_t num_blocks = composite->at(i).size();
    const size_t block_size = sub_managers_[i]->block_size();
    const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
    if (num_blocks_needed <= num_blocks) {
      continue;
    }

    const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

    const auto blocks = sub_managers_[i]->allocate(num_additional_blocks);
    if (blocks.size() != num_additional_blocks) {
      return false;
    }

    composite->at(i).insert(
        composite->at(i).end(), blocks.begin(), blocks.end());
  }
  return true;
}

void CompositeBlockManager::deallocate_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  const std::vector<std::vector<Block>>& composite =
      seq->kv_state().composite_blocks();
  for (size_t i = 0; i < sub_managers_.size(); ++i) {
    if (!composite[i].empty()) {
      sub_managers_[i]->deallocate(composite[i]);
    }
  }
}

void CompositeBlockManager::deallocate(const Slice<Block>& blocks) {
  LOG(FATAL) << "CompositeBlockManager::deallocate is not implemented";
}

std::vector<Block> CompositeBlockManager::allocate(size_t num_blocks) {
  LOG(FATAL) << "CompositeBlockManager::allocate is not implemented";
  return {};
}

std::vector<Block> CompositeBlockManager::allocate_shared(
    const Slice<int32_t>& tokens_ids,
    const Slice<Block>& existed_shared_blocks) {
  LOG(FATAL) << "CompositeBlockManager::allocate_shared is not implemented";
  return {};
}

void CompositeBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                  std::vector<Block>& /*blocks*/,
                                  size_t /*existed_shared_blocks_num*/) {
  // Prefix cache is not supported for CompositeBlockManager yet.
  // Keep cache() as a no-op so scheduler-side prefill cache hooks
  // do not crash when composite manager is enabled (e.g. DeepSeek V4).
}

void CompositeBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  // Prefix cache is not supported for CompositeBlockManager yet.
}

void CompositeBlockManager::get_merged_kvcache_event(
    KvCacheEvent* event) const {
  for (const auto& mgr : sub_managers_) {
    mgr->get_merged_kvcache_event(event);
  }
}

size_t CompositeBlockManager::num_blocks_in_prefix_cache() const {
  LOG(FATAL) << "CompositeBlockManager not support prefix cache yet";
  return 0;
}

size_t CompositeBlockManager::num_free_blocks() const {
  return sub_managers_[1]->num_free_blocks();
}

size_t CompositeBlockManager::num_used_blocks() const {
  return sub_managers_[1]->num_used_blocks();
}

double CompositeBlockManager::kv_cache_utilization() const {
  return sub_managers_[1]->kv_cache_utilization();
}

void CompositeBlockManager::free(int32_t block_id) {
  LOG(FATAL) << "CompositeBlockManager::free should not be called";
}

Block CompositeBlockManager::allocate() {
  LOG(FATAL) << "CompositeBlockManager::allocate should not be called";
  return Block();
}

size_t CompositeBlockManager::num_total_blocks() const {
  return sub_managers_[1]->num_total_blocks();
}

}  // namespace xllm
