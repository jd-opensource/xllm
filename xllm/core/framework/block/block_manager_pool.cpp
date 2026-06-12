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

#include "block_manager_pool.h"

#include <algorithm>
#include <limits>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/block/cache_group_spec_builder.h"
#include "framework/request/sequence.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"

namespace xllm {
namespace {

// Single C1 cache group describing the normal-model composite manager: one
// incremental attention pool sized from the pool options, prefix-cacheable only
// when prefix caching is enabled. SINGLE_RES stays on the legacy
// single_block_managers_ during phase 1, so it is intentionally absent here.
std::vector<CacheGroupSpec> make_normal_composite_specs(
    const BlockManagerPool::Options& options) {
  CacheGroupSpec c1;
  c1.state_id = CacheStateId::C1;
  c1.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  c1.block_size = static_cast<uint32_t>(options.block_size());
  c1.num_blocks = options.num_blocks();
  c1.prefix_cacheable = options.enable_prefix_cache();
  c1.prefix_group = options.enable_prefix_cache() ? PrefixCacheGroup::C1
                                                  : PrefixCacheGroup::INVALID;
  return {c1};
}

// DeepSeek-V4 cache-group composite: a rolling-window SWA attention group
// followed by one incremental compressed group per compress ratio (C4, C128),
// in worker multi_block_tables export order. The trailing SINGLE_RES spec is
// stripped -- DSV4's per-sequence resource block stays on the legacy
// single_block_managers_ exactly like the normal path. Allocator sizing
// preserves the original DSV4 cache-state pool formula:
//   - SWA shares a burst pool across the in-flight batch.
//   - Each compressed pool shrinks by its ratio (one compressed block covers
//     `ratio` base blocks' worth of tokens).
// The SWA ring exports exactly `swa_window_blocks` columns, which is the modulo
// base the worker reads as semantic_cols = raw_bt.size(1)
// (dsa_metadata_builder.cpp), so no worker change is needed.
std::vector<CacheGroupSpec> make_dsv4_composite_specs(
    const BlockManagerPool::Options& options) {
  const uint32_t block_size = static_cast<uint32_t>(options.block_size());
  CHECK_GT(block_size, 0u) << "block_size must be positive for DSV4";
  const uint32_t swa_blocks_per_seq = options.swa_blocks_per_seq();
  CHECK_GT(swa_blocks_per_seq, 0u)
      << "swa_blocks_per_seq must be positive for DSV4";

  ModelCacheGroupConfig config;
  config.base_block_size = block_size;
  // SINGLE_RES is stripped below, so its sizing here is irrelevant.
  config.single_res_num_blocks = 0;

  const uint32_t max_seqs = std::max(options.max_seqs_per_batch(), 1u);
  const uint32_t burst_blocks =
      (std::max(options.max_tokens_per_batch(), 1u) + block_size - 1) /
      block_size;
  config.swa_window_blocks = swa_blocks_per_seq;
  config.swa_num_blocks =
      swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;

  // The engine prepends a placeholder ratio (0) for the SWA slot; keep only the
  // real compressed ratios (C4=4, C128=128).
  for (const uint32_t ratio : options.compress_ratios()) {
    if (ratio <= 1) {
      continue;
    }
    config.compress_ratios.push_back(ratio);
    config.compress_num_blocks.push_back(options.num_blocks() / ratio);
  }

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);
  CHECK(!specs.empty() && specs.back().state_id == CacheStateId::SINGLE_RES)
      << "build_cache_group_specs must append SINGLE_RES last";
  specs.pop_back();
  return specs;
}

// Bind a sequence's device KV state to one composite manager call. The normal
// pool only ever targets the device role; the host pool is unaffected here.
// The token view and prefix-hash chain let the composite insert completed
// blocks into its prefix caches from inside allocate/deallocate (see
// GroupCompositeBlockManager::insert_committed_blocks). Callers that must
// release without caching (preempt) use make_bare_device_context instead.
BlockManagerContext make_device_context(Sequence* sequence, int32_t dp_rank) {
  BlockManagerContext context;
  context.sequence = sequence;
  context.kv_state = &sequence->kv_state();
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = dp_rank;
  context.tokens = sequence->tokens();
  context.hash_state = &sequence->prefix_hash_state();
  return context;
}

// Like make_device_context but without the token view / hash chain, so the
// composite's internal prefix-cache insert is skipped. Used by the preempt
// path, which must release uncomputed blocks without writing them to a cache.
BlockManagerContext make_bare_device_context(Sequence* sequence,
                                             int32_t dp_rank) {
  BlockManagerContext context;
  context.sequence = sequence;
  context.kv_state = &sequence->kv_state();
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = dp_rank;
  return context;
}

}  // namespace

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  single_block_managers_.reserve(dp_size);

  BlockManager::Options block_options;
  block_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .sliding_window_size(options_.sliding_window_size())
      .swa_blocks_per_seq(options_.swa_blocks_per_seq())
      .max_tokens_per_batch(options_.max_tokens_per_batch())
      .manager_types(options_.manager_types())
      .compress_ratios(options_.compress_ratios())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .hasher_type(options_.hasher_type());

  const uint32_t max_single_block_sequences =
      options_.max_concurrent_requests() > 0
          ? options_.max_concurrent_requests()
          : static_cast<uint32_t>(std::max(
                ::xllm::ServiceConfig::get_instance().max_concurrent_requests(),
                0));
  const uint32_t num_single_blocks = std::max<uint32_t>(
      options_.num_single_blocks(), max_single_block_sequences + 2);
  CHECK_GT(num_single_blocks, 0u) << "num_single_blocks must be positive";

  // DSV4 (SWA + compressed) is selected by a non-empty manager_types and always
  // routes through the cache-group composite, in every mode. The normal LLM
  // path routes through the composite only on the plain device role; its
  // specialized modes (disagg-PD / kvcache-store / host blocks) stay on
  // ConcurrentBlockManagerImpl. xtensor is handled separately and never
  // composite.
  const bool is_dsv4 = !options_.manager_types().empty();
  const bool normal_device_path =
      !options_.enable_disagg_pd() && !options_.enable_kvcache_store() &&
      !options_.enable_host_blocks() && options_.host_num_blocks() == 0;
  composite_ =
      !options_.enable_xtensor() && (is_dsv4 || normal_device_path);
  if (composite_) {
    composite_managers_.reserve(dp_size);
  }

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options_.enable_xtensor()) {
      // Use XTensorBlockManagerImpl for xtensor mode.
      CHECK_GT(options_.num_layers(), 0)
          << "num_layers must be set when enable_xtensor is true";
      CHECK_GT(options_.slot_size(), 0)
          << "slot_size must be set when enable_xtensor is true";
      size_t page_size =
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
      // In the current implementation, K and V must be the same size, so we
      // divide by 2.
      size_t block_mem_size =
          static_cast<size_t>(options_.block_size()) * options_.slot_size() / 2;
      block_managers_.emplace_back(
          std::make_unique<XTensorBlockManagerImpl>(block_options,
                                                    options_.num_layers(),
                                                    block_mem_size,
                                                    page_size,
                                                    /*dp_rank=*/i,
                                                    options_.model_id()));
    } else if (composite_) {
      // One composite manager per DP rank: the DSV4 SWA/C4/C128 groups, or the
      // normal single-C1 group.
      composite_managers_.emplace_back(
          std::make_unique<ConcurrentCompositeBlockManager>(
              is_dsv4 ? make_dsv4_composite_specs(options_)
                      : make_normal_composite_specs(options_)));
    } else {
      // Normal model in a specialized mode (disagg-PD / kvcache-store / host).
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(block_options));
    }
    // Scheduler-side per-sequence resources share one logical single-block
    // pool. Worker-side embedding and linear-state caches remain physically
    // separate and are addressed via transport fields.
    single_block_managers_.emplace_back(std::make_unique<SingleBlockManager>(
        /*num_blocks=*/num_single_blocks,
        /*resource_name=*/"single block",
        /*exhaustion_message=*/"No more single-block ids available"));
  }
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(manager_count());
}

size_t BlockManagerPool::manager_count() const {
  return composite_ ? composite_managers_.size()
                           : block_managers_.size();
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  const size_t count = manager_count();
  if (count == 0) {
    return 0;
  }

  auto free_blocks_at = [this](size_t i) -> size_t {
    return composite_ ? composite_managers_[i]->num_free_blocks()
                             : block_managers_[i]->num_free_blocks();
  };

  size_t max_index = 0;
  size_t max_free = free_blocks_at(0);
  for (size_t i = 1; i < count; ++i) {
    const size_t current_free = free_blocks_at(i);
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

bool BlockManagerPool::allocate_single_block(Sequence* sequence,
                                             int32_t dp_rank) {
  CHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  if (sequence->has_single_block_id()) {
    return true;
  }

  auto single_blocks = single_block_managers_[dp_rank]->allocate(1);
  if (single_blocks.empty()) {
    const auto* manager = single_block_managers_[dp_rank].get();
    LOG(ERROR) << "Failed to allocate single block! dp_rank=" << dp_rank
               << ", free=" << manager->num_free_blocks()
               << ", used=" << manager->num_used_blocks()
               << ", total=" << manager->num_total_blocks()
               << ", max_seqs_per_batch=" << options_.max_seqs_per_batch()
               << ", configured_single_blocks=" << options_.num_single_blocks();
    return false;
  }
  sequence->set_single_block(std::move(single_blocks[0]));
  return true;
}

void BlockManagerPool::deallocate_single_block(Sequence* sequence,
                                               int32_t dp_rank) {
  DCHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  auto single_block = sequence->reset_single_block();
  if (!single_block.is_valid()) {
    return;
  }
  single_block_managers_[dp_rank]->deallocate({&single_block, 1});
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    deallocate(sequence);
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  if (composite_) {
    // The composite inserts the final completed blocks into the prefix cache
    // from inside deallocate (the context carries the token view + hash chain),
    // then releases every group.
    BlockManagerContext context = make_device_context(sequence, dp_rank);
    composite_managers_[dp_rank]->deallocate(&context);
    deallocate_single_block(sequence, dp_rank);
    sequence->reset();
    return;
  }

  // add blocks to the prefix cache
  cache(sequence);
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_single_block(sequence, dp_rank);
  // release the blocks after prefix cache insertion
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

bool BlockManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence, sequence->num_tokens())) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  const bool started_empty = sequence->kv_state().num_kv_blocks() == 0;
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    return false;
  }

  if (composite_) {
    if (options_.enable_prefix_cache() && started_empty) {
      // First prefill: restore any cached prefix before growing; growth
      // appends on top of the matched C1 blocks. The lazy flush for subsequent
      // grows (decode / chunked prefill) now happens inside the composite's
      // allocate, so the scheduler never drives a separate flush.
      composite_match_shared(sequence, dp_rank);
    }
    BlockManagerContext context = make_device_context(sequence, dp_rank);
    if (!composite_managers_[dp_rank]->allocate(&context, num_tokens)) {
      // Only a fresh sequence is fully unwound; an in-flight decode keeps its
      // existing blocks for the scheduler to preempt (matches the legacy path).
      if (started_empty) {
        composite_managers_[dp_rank]->deallocate(&context);
        if (needs_single_block) {
          deallocate_single_block(sequence, dp_rank);
        }
        sequence->reset();
      }
      return false;
    }
    return true;
  }

  // first try to allocate shared blocks
  if (started_empty) {
    allocate_shared(sequence);
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    return process_beam_search(sequence, /*need_swap*/ true);
  }
  process_beam_search(sequence);

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    if (started_empty) {
      block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      sequence->reset();
    }
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";

    return false;
  }

  sequence->add_kv_blocks(blocks);

  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  // Raw block vectors have no per-sequence group state, so they have no meaning
  // on the composite path. No caller hits this on the normal path today.
  CHECK(!composite_)
      << "raw allocate(num_tokens, dp_rank) is unsupported on the composite "
         "path";
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    return false;
  }

  if (composite_) {
    BlockManagerContext context = make_device_context(sequence, dp_rank);
    if (options_.enable_prefix_cache()) {
      composite_managers_[dp_rank]->match_prefix_cache(
          &context, sequence->tokens(), &sequence->mm_data());
    }
    if (!composite_managers_[dp_rank]->allocate(&context,
                                                sequence->tokens().size())) {
      composite_managers_[dp_rank]->deallocate(&context);
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      sequence->reset();
      return false;
    }
    // try_allocate reserves the whole prompt: mark every token as cached, the
    // same end-state the legacy path reaches via incr_kv_cache_tokens_num.
    sequence->kv_state().set_kv_cache_tokens_num(sequence->tokens().size());
    return true;
  }

  std::vector<Block> shared_blocks;
  size_t shared_num = 0;
  if (options_.enable_prefix_cache()) {
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    shared_blocks = block_managers_[dp_rank]->allocate_shared(
        sequence->tokens(), existed_shared_blocks, sequence->mm_data());

    if (!shared_blocks.empty()) {
      sequence->add_kv_blocks(shared_blocks);
      sequence->kv_state().incr_shared_kv_blocks_num(shared_blocks.size());
      shared_num = shared_blocks.size();
    }
  }

  const size_t block_size = options_.block_size();
  size_t num_tokens = sequence->tokens().size() - shared_num * block_size;

  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > 0) {
    const auto blocks = block_managers_[dp_rank]->allocate(num_blocks_needed);
    if (blocks.size() != num_blocks_needed) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      if (shared_num != 0) {
        block_managers_[dp_rank]->deallocate(shared_blocks);
        sequence->reset();
      }
      return false;
    }

    sequence->add_kv_blocks(std::move(blocks));
  }

  sequence->kv_state().incr_kv_cache_tokens_num(sequence->tokens().size());
  return true;
}

bool BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return true;
  }

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    auto new_blocks = block_managers_[dp_rank]->allocate(1);
    if (new_blocks.size() == 0) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks[0]);
  } else {
    sequence->kv_state().process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  if (composite_) {
    composite_match_shared(sequence, dp_rank);
    return;
  }
  const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
      0, sequence->kv_state().shared_kv_blocks_num());
  // If the sequence holds shared_blocks, the hash values of these blocks do
  // not need to be recalculated and can be reused directly.
  std::vector<Block> shared_blocks = block_managers_[dp_rank]->allocate_shared(
      sequence->tokens(), existed_shared_blocks, sequence->mm_data());
  sequence->add_shared_kv_blocks(std::move(shared_blocks));
}

void BlockManagerPool::composite_match_shared(Sequence* sequence,
                                              int32_t dp_rank) {
  BlockManagerContext context = make_device_context(sequence, dp_rank);
  CompositeMatchResult matched =
      composite_managers_[dp_rank]->match_prefix_cache(
          &context, sequence->tokens(), &sequence->mm_data());
  if (matched.matched_tokens == 0) {
    return;
  }

  size_t matched_tokens = matched.matched_tokens;
  const size_t total_tokens = sequence->num_tokens();
  // Whole-prompt cache hit: drop the last shared block so the forward pass has
  // at least one token to (re)compute. Mirrors KVCacheState::add_shared_kv_-
  // blocks, which pops the last block and rewinds the cached-token position.
  if (matched_tokens >= total_tokens) {
    CacheGroupState* c1 = sequence->kv_state().group_state(CacheStateId::C1);
    if (c1 != nullptr && !c1->blocks.empty()) {
      const size_t block_size = c1->blocks.back().size();
      c1->blocks.pop_back();
      c1->shared_blocks_num = c1->blocks.size();
      c1->prefix_cached_tokens = c1->blocks.size() * block_size;
      matched_tokens = c1->prefix_cached_tokens;
    }
  }
  sequence->kv_state().set_kv_cache_tokens_num(matched_tokens);
}

void BlockManagerPool::cache(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  if (composite_) {
    // The composite inserts completed blocks into its prefix caches internally
    // from allocate/deallocate; the scheduler no longer drives a separate
    // cache step on this path.
    return;
  }
  const auto token_ids = sequence->cached_tokens();
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  auto existed_shared_blocks_num = sequence->kv_state().shared_kv_blocks_num();
  block_managers_[dp_rank]->cache(
      token_ids, *blocks, existed_shared_blocks_num, sequence->mm_data());
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  const size_t count = manager_count();
  if (count == 0) {
    return 0.0f;
  }
  float perc = 0.0;
  for (size_t i = 0; i < count; ++i) {
    perc += composite_ ? composite_managers_[i]->kv_cache_utilization()
                              : block_managers_[i]->kv_cache_utilization();
  }
  return perc / count;
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(manager_count());
  if (!options_.enable_prefix_cache()) {
    return num_blocks_in_prefix_cache;
  }
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    if (composite_) {
      num_blocks_in_prefix_cache[dp_rank] =
          composite_managers_[dp_rank]->num_blocks_in_prefix_cache();
      continue;
    }
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(manager_count());
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    num_free_blocks[dp_rank] =
        composite_ ? composite_managers_[dp_rank]->num_free_blocks()
                          : block_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(manager_count());
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    num_used_blocks[dp_rank] =
        composite_ ? composite_managers_[dp_rank]->num_used_blocks()
                          : block_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return composite_ ? composite_managers_[dp_rank]->kv_cache_utilization()
                    : block_managers_[dp_rank]->kv_cache_utilization();
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  if (composite_) {
    // Release every group without inserting into the prefix cache: a bare
    // context carries no token view / hash chain, so the composite's internal
    // insert is skipped and uncomputed blocks are never cached.
    BlockManagerContext context = make_bare_device_context(sequence, dp_rank);
    composite_managers_[dp_rank]->deallocate(&context);
    deallocate_single_block(sequence, dp_rank);
    sequence->reset();
    return;
  }

  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_single_block(sequence, dp_rank);
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }

  // Reserve padding block on each XTensorBlockManagerImpl.
  for (auto& manager : block_managers_) {
    auto* xtensor_manager =
        dynamic_cast<XTensorBlockManagerImpl*>(manager.get());
    if (xtensor_manager) {
      xtensor_manager->reserve_xtensor_padding_blocks();
    }
  }

  // Start prealloc thread once (PageAllocator is shared by all managers)
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm
