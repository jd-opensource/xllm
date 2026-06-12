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

#include "framework/block/cache_group_spec_builder.h"

#include <glog/logging.h>

namespace xllm {
namespace {

// The universal per-sequence resource group, shared by every model shape. One
// group, two worker exports (linear_state_ids + embedding_ids), no prefix
// cache. Appended last so it never disturbs the block_tables / multi_block_-
// tables export order of the attention groups.
CacheGroupSpec make_single_res_spec(uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SINGLE_RES;
  spec.policy_type = CachePolicyType::PER_SEQUENCE_ONCE;
  spec.prefix_group = PrefixCacheGroup::INVALID;
  spec.block_size = 1;
  spec.num_blocks = num_blocks;
  spec.compress_ratio = 1;
  spec.prefix_cacheable = false;
  spec.export_targets = {WorkerExportTarget::LINEAR_STATE_IDS,
                         WorkerExportTarget::EMBEDDING_IDS};
  spec.export_index = -1;
  return spec;
}

CacheGroupSpec make_c1_spec(uint32_t base_block_size,
                            uint32_t num_blocks,
                            bool prefix_cacheable) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::C1;
  spec.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  spec.prefix_group =
      prefix_cacheable ? PrefixCacheGroup::C1 : PrefixCacheGroup::INVALID;
  spec.block_size = base_block_size;
  spec.num_blocks = num_blocks;
  spec.compress_ratio = 1;
  spec.prefix_cacheable = prefix_cacheable;
  spec.export_targets = {WorkerExportTarget::BLOCK_TABLES};
  spec.export_index = -1;
  return spec;
}

CacheGroupSpec make_swa_spec(uint32_t base_block_size,
                             uint32_t window_blocks,
                             uint32_t num_blocks,
                             int32_t export_index) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SWA;
  spec.policy_type = CachePolicyType::ROLLING_WINDOW;
  // Target form uses PrefixCacheGroup::SINGLE; phase 1 keeps it non-cacheable.
  spec.prefix_group = PrefixCacheGroup::SINGLE;
  spec.block_size = base_block_size;
  spec.num_blocks = num_blocks;
  spec.compress_ratio = 1;
  spec.window_blocks = window_blocks;
  spec.prefix_cacheable = false;
  spec.export_targets = {WorkerExportTarget::MULTI_BLOCK_TABLES};
  spec.export_index = export_index;
  return spec;
}

CacheGroupSpec make_compressed_spec(uint32_t base_block_size,
                                    uint32_t compress_ratio,
                                    uint32_t num_blocks,
                                    int32_t export_index) {
  CacheStateId state_id = CacheStateId::C4;
  PrefixCacheGroup prefix_group = PrefixCacheGroup::C4;
  if (compress_ratio == 4) {
    state_id = CacheStateId::C4;
    prefix_group = PrefixCacheGroup::C4;
  } else if (compress_ratio == 128) {
    state_id = CacheStateId::C128;
    prefix_group = PrefixCacheGroup::C128;
  } else {
    LOG(FATAL) << "unsupported compress ratio " << compress_ratio
               << "; phase-1 supports 4 (C4) and 128 (C128) only";
  }

  CacheGroupSpec spec;
  spec.state_id = state_id;
  spec.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  spec.prefix_group = prefix_group;
  spec.block_size = base_block_size * compress_ratio;
  spec.num_blocks = num_blocks;
  spec.compress_ratio = compress_ratio;
  // DSV4 compressed-group prefix caching is deferred to a later phase.
  spec.prefix_cacheable = false;
  spec.export_targets = {WorkerExportTarget::MULTI_BLOCK_TABLES};
  spec.export_index = export_index;
  return spec;
}

}  // namespace

std::vector<CacheGroupSpec> build_cache_group_specs(
    const ModelCacheGroupConfig& config) {
  CHECK_GT(config.base_block_size, 0u) << "base_block_size must be positive";
  CHECK_EQ(config.compress_ratios.size(), config.compress_num_blocks.size())
      << "compress_ratios and compress_num_blocks must be index-aligned";

  const bool is_dsv4 =
      config.swa_window_blocks > 0 || !config.compress_ratios.empty();

  std::vector<CacheGroupSpec> specs;
  if (is_dsv4) {
    CHECK(!config.has_linear_state)
        << "DSV4 (SWA/compressed) and linear-state shapes are mutually "
           "exclusive";
    // Worker export order is the multi_block_tables index order: SWA[0],
    // then each compressed group in declared ratio order.
    int32_t next_export_index = 0;
    if (config.swa_window_blocks > 0) {
      specs.push_back(make_swa_spec(config.base_block_size,
                                    config.swa_window_blocks,
                                    config.swa_num_blocks,
                                    next_export_index++));
    }
    for (size_t i = 0; i < config.compress_ratios.size(); ++i) {
      specs.push_back(make_compressed_spec(config.base_block_size,
                                           config.compress_ratios[i],
                                           config.compress_num_blocks[i],
                                           next_export_index++));
    }
  } else {
    // Normal and Qwen3.5+ share the C1 + SINGLE_RES shape; they differ only in
    // whether C1 participates in prefix caching.
    specs.push_back(
        make_c1_spec(config.base_block_size,
                     config.c1_num_blocks,
                     /*prefix_cacheable=*/!config.has_linear_state));
  }

  specs.push_back(make_single_res_spec(config.single_res_num_blocks));

  // A linear-state model must never expose a cacheable group (see doc: a C1
  // hit would strand the unrecoverable recurrent state).
  if (config.has_linear_state) {
    for (const CacheGroupSpec& spec : specs) {
      CHECK(!spec.prefix_cacheable)
          << "linear-state model must not declare any prefix-cacheable group ("
          << to_string(spec.state_id) << ")";
    }
  }

  return specs;
}

}  // namespace xllm
