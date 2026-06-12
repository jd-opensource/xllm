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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/block/cache_group.h"

namespace xllm {
namespace {

const CacheGroupSpec* find_spec(const std::vector<CacheGroupSpec>& specs,
                                CacheStateId state_id) {
  for (const CacheGroupSpec& spec : specs) {
    if (spec.state_id == state_id) {
      return &spec;
    }
  }
  return nullptr;
}

}  // namespace

// Normal model: a prefix-cacheable C1 group followed by SINGLE_RES.
TEST(CacheGroupSpecBuilderTest, NormalModelEmitsCacheableC1AndSingleRes) {
  ModelCacheGroupConfig config;
  config.base_block_size = 16;
  config.c1_num_blocks = 100;
  config.single_res_num_blocks = 50;

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);

  ASSERT_EQ(specs.size(), 2u);
  EXPECT_EQ(specs[0].state_id, CacheStateId::C1);
  EXPECT_EQ(specs[1].state_id, CacheStateId::SINGLE_RES);

  const CacheGroupSpec& c1 = specs[0];
  EXPECT_EQ(c1.policy_type, CachePolicyType::INCREMENTAL_APPEND);
  EXPECT_EQ(c1.prefix_group, PrefixCacheGroup::C1);
  EXPECT_EQ(c1.block_size, 16u);
  EXPECT_EQ(c1.num_blocks, 100u);
  EXPECT_EQ(c1.compress_ratio, 1u);
  EXPECT_TRUE(c1.prefix_cacheable);
  ASSERT_EQ(c1.export_targets.size(), 1u);
  EXPECT_EQ(c1.export_targets[0], WorkerExportTarget::BLOCK_TABLES);
}

// The universal SINGLE_RES group is always present, last, and never cacheable.
TEST(CacheGroupSpecBuilderTest, SingleResIsUniversalAndNonCacheable) {
  ModelCacheGroupConfig config;
  config.base_block_size = 16;
  config.c1_num_blocks = 10;
  config.single_res_num_blocks = 42;

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);

  const CacheGroupSpec& single_res = specs.back();
  EXPECT_EQ(single_res.state_id, CacheStateId::SINGLE_RES);
  EXPECT_EQ(single_res.policy_type, CachePolicyType::PER_SEQUENCE_ONCE);
  EXPECT_EQ(single_res.prefix_group, PrefixCacheGroup::INVALID);
  EXPECT_EQ(single_res.block_size, 1u);
  EXPECT_EQ(single_res.num_blocks, 42u);
  EXPECT_FALSE(single_res.prefix_cacheable);
  ASSERT_EQ(single_res.export_targets.size(), 2u);
  EXPECT_EQ(single_res.export_targets[0], WorkerExportTarget::LINEAR_STATE_IDS);
  EXPECT_EQ(single_res.export_targets[1], WorkerExportTarget::EMBEDDING_IDS);
}

// Qwen3.5+ hybrid: C1 is present but non-cacheable (linear state cannot be
// restored from a prefix hit), and no group is cacheable.
TEST(CacheGroupSpecBuilderTest, LinearStateModelDisablesPrefixCaching) {
  ModelCacheGroupConfig config;
  config.base_block_size = 16;
  config.c1_num_blocks = 100;
  config.single_res_num_blocks = 50;
  config.has_linear_state = true;

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);

  ASSERT_EQ(specs.size(), 2u);
  const CacheGroupSpec* c1 = find_spec(specs, CacheStateId::C1);
  ASSERT_NE(c1, nullptr);
  EXPECT_FALSE(c1->prefix_cacheable);
  EXPECT_EQ(c1->prefix_group, PrefixCacheGroup::INVALID);

  for (const CacheGroupSpec& spec : specs) {
    EXPECT_FALSE(spec.prefix_cacheable);
  }
}

// DSV4: SWA (rolling window) then compressed C4/C128 in multi_block_tables
// export order, then SINGLE_RES. No C1, nothing cacheable in phase 1.
TEST(CacheGroupSpecBuilderTest, Dsv4EmitsSwaAndCompressedGroupsInExportOrder) {
  ModelCacheGroupConfig config;
  config.base_block_size = 16;
  config.single_res_num_blocks = 50;
  config.swa_window_blocks = 8;
  config.swa_num_blocks = 64;
  config.compress_ratios = {4, 128};
  config.compress_num_blocks = {40, 10};

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);

  ASSERT_EQ(specs.size(), 4u);
  EXPECT_EQ(specs[0].state_id, CacheStateId::SWA);
  EXPECT_EQ(specs[1].state_id, CacheStateId::C4);
  EXPECT_EQ(specs[2].state_id, CacheStateId::C128);
  EXPECT_EQ(specs[3].state_id, CacheStateId::SINGLE_RES);
  // SINGLE_RES is not exported to multi_block_tables (sentinel export_index),
  // so multi_block_table_groups() skips it for DSV4 sequences.
  EXPECT_EQ(specs[3].export_index, -1);

  EXPECT_EQ(find_spec(specs, CacheStateId::C1), nullptr);

  const CacheGroupSpec& swa = specs[0];
  EXPECT_EQ(swa.policy_type, CachePolicyType::ROLLING_WINDOW);
  EXPECT_EQ(swa.block_size, 16u);
  // The rolling-window SWA group exports exactly window_blocks columns. That
  // column count is the modulo base the worker reads as
  // semantic_cols = raw_bt.size(1) (dsa_metadata_builder.cpp:414) and indexes as
  // (pos / block_size) % semantic_cols. So exporting the ring (window_blocks
  // wide) is equivalent to the OLD full-length-with-holes SWA table -- the
  // kernel never touches columns outside the window -- and needs no worker edit.
  EXPECT_EQ(swa.window_blocks, 8u);
  EXPECT_EQ(swa.num_blocks, 64u);
  EXPECT_EQ(swa.export_targets[0], WorkerExportTarget::MULTI_BLOCK_TABLES);
  EXPECT_EQ(swa.export_index, 0);

  const CacheGroupSpec& c4 = specs[1];
  EXPECT_EQ(c4.block_size, 16u * 4u);
  EXPECT_EQ(c4.compress_ratio, 4u);
  EXPECT_EQ(c4.num_blocks, 40u);
  EXPECT_EQ(c4.export_index, 1);

  const CacheGroupSpec& c128 = specs[2];
  EXPECT_EQ(c128.block_size, 16u * 128u);
  EXPECT_EQ(c128.compress_ratio, 128u);
  EXPECT_EQ(c128.num_blocks, 10u);
  EXPECT_EQ(c128.export_index, 2);

  for (const CacheGroupSpec& spec : specs) {
    EXPECT_FALSE(spec.prefix_cacheable);
  }
}

// Compressed groups without an SWA group still export from index 0 upward.
TEST(CacheGroupSpecBuilderTest, CompressedOnlyConfigStartsExportIndexAtZero) {
  ModelCacheGroupConfig config;
  config.base_block_size = 32;
  config.single_res_num_blocks = 8;
  config.compress_ratios = {4};
  config.compress_num_blocks = {20};

  std::vector<CacheGroupSpec> specs = build_cache_group_specs(config);

  ASSERT_EQ(specs.size(), 2u);
  EXPECT_EQ(specs[0].state_id, CacheStateId::C4);
  EXPECT_EQ(specs[0].export_index, 0);
  EXPECT_EQ(specs[1].state_id, CacheStateId::SINGLE_RES);
}

}  // namespace xllm
