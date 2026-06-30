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

#include "kv_cache.h"

#include <gtest/gtest.h>

#include <vector>

#include "framework/block/block.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "kv_cache_shape.h"
#include "platform/device.h"

namespace xllm {

namespace {

std::vector<int64_t> shape_vec(const torch::Tensor& tensor) {
  return tensor.sizes().vec();
}

std::vector<int64_t> dsv4_block_shape(int64_t block_count,
                                      int64_t block_size,
                                      int64_t n_heads,
                                      int64_t head_dim) {
#if defined(USE_MLU)
  return {block_count, n_heads, block_size, head_dim};
#else
  return {block_count, block_size, n_heads, head_dim};
#endif
}

}  // namespace

// Host prefix-cache allocation registers page-aligned host memory with the NPU
// via aclrtHostRegister, which requires a live device context. Set one up once.
class HostKVCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
  }
};

TEST(KVCacheTest, DeepSeekV4FourDimCachesUseDeviceLayout) {
  constexpr int64_t kSwaCount = 10;
  constexpr int64_t kC4Count = 32;
  constexpr int64_t kC128Count = 1;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 16;
  constexpr int64_t kIndexHeadDim = 8;

  KVCacheCapacity capacity;
  capacity.block_size(kBlockSize)
      .swa_count(kSwaCount)
      .c4_count(kC4Count)
      .c128_count(kC128Count);

  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(3)
      .model_type("deepseek_v4")
      .block_size(kBlockSize)
      .head_dim(kHeadDim)
      .index_head_dim(kIndexHeadDim)
      .window_size(/*window_size=*/512)
      .compress_ratios({1, 4, 128});

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  ASSERT_EQ(caches.size(), 3u);

  EXPECT_EQ(shape_vec(caches[0].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  EXPECT_FALSE(caches[0].get_compress_kv_state().defined());

  EXPECT_EQ(shape_vec(caches[1].get_k_cache()),
            dsv4_block_shape(kC4Count, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[1].get_index_cache()),
            dsv4_block_shape(kC4Count, kBlockSize, 1, kIndexHeadDim));
  EXPECT_EQ(shape_vec(caches[1].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  if (caches[1].get_indexer_cache_scale().defined()) {
    EXPECT_EQ(shape_vec(caches[1].get_indexer_cache_scale()),
              (std::vector<int64_t>{kC4Count, kBlockSize, 1}));
  }
  EXPECT_EQ(shape_vec(caches[1].get_compress_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_index_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kIndexHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_index_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kIndexHeadDim}));

  EXPECT_EQ(shape_vec(caches[2].get_k_cache()),
            dsv4_block_shape(kC128Count, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_compress_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
  EXPECT_EQ(shape_vec(caches[2].get_compress_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
}

TEST_F(HostKVCacheTest, HostKVCacheNormalLayoutAddsLayerDim) {
  constexpr int64_t kNumBlocks = 16;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 64;
  constexpr int64_t kNumHeads = 4;
  constexpr int64_t kLayerCount = 5;
  constexpr double kHostFactor = 2.0;

  KVCacheCapacity capacity;
  capacity.n_blocks(kNumBlocks).block_size(kBlockSize);

  ModelArgs model_args;
  model_args.model_type("qwen");
  model_args.n_kv_heads(kNumHeads);
  model_args.head_dim(kHeadDim);
  KVCacheShape shape(capacity, model_args, /*world_size=*/1);
  ASSERT_TRUE(shape.has_key_cache_shape());

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("qwen")
      .host_blocks_factor(kHostFactor);

  KVCache host_cache(shape, options, BlockType::KV, kLayerCount);

  const BlockTypeTensorMap tensors =
      host_cache.get_block_type_tensors(BlockType::KV);
  ASSERT_TRUE(tensors.count(KVCacheTensorRole::KEY) > 0);

  const std::vector<int64_t> base_key_shape = shape.key_cache_shape();
  const torch::Tensor& host_key = tensors.at(KVCacheTensorRole::KEY);
  EXPECT_TRUE(host_key.is_contiguous());
  EXPECT_EQ(host_key.device().type(), torch::kCPU);
  // host shape == [scaled_blocks, layer_count, ...per_block_dims]
  ASSERT_EQ(host_key.dim(), static_cast<int64_t>(base_key_shape.size()) + 1);
  EXPECT_EQ(host_key.size(0),
            scale_host_block_count(base_key_shape[0], kHostFactor));
  EXPECT_EQ(host_key.size(1), kLayerCount);
  for (size_t i = 1; i < base_key_shape.size(); ++i) {
    EXPECT_EQ(host_key.size(static_cast<int64_t>(i) + 1), base_key_shape[i]);
  }
}

TEST_F(HostKVCacheTest, HostKVCacheDeepSeekV4PerBlockType) {
  constexpr int64_t kSwaCount = 10;
  constexpr int64_t kC4Count = 32;
  constexpr int64_t kC128Count = 4;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 16;
  constexpr int64_t kIndexHeadDim = 8;
  constexpr double kHostFactor = 3.0;

  KVCacheCapacity capacity;
  capacity.block_size(kBlockSize)
      .swa_count(kSwaCount)
      .c4_count(kC4Count)
      .c128_count(kC128Count);

  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(3)
      .model_type("deepseek_v4")
      .block_size(kBlockSize)
      .head_dim(kHeadDim)
      .index_head_dim(kIndexHeadDim)
      .window_size(/*window_size=*/512)
      .compress_ratios({1, 4, 128})
      .host_blocks_factor(kHostFactor);

  // SWA host cache: 1 layer in this 3-layer config (compress_ratio == 1).
  KVCache swa_host(shape, options, BlockType::SWA, /*layer_count=*/1);
  const BlockTypeTensorMap swa_tensors =
      swa_host.get_block_type_tensors(BlockType::SWA);
  ASSERT_TRUE(swa_tensors.count(KVCacheTensorRole::SWA) > 0);
  const torch::Tensor& swa = swa_tensors.at(KVCacheTensorRole::SWA);
  EXPECT_TRUE(swa.is_contiguous());
  EXPECT_EQ(swa.size(0), scale_host_block_count(kSwaCount, kHostFactor));
  EXPECT_EQ(swa.size(1), 1);

  // C4 host cache: key + index, index uses the DSV4 index dtype.
  KVCache c4_host(shape, options, BlockType::C4, /*layer_count=*/1);
  const BlockTypeTensorMap c4_tensors =
      c4_host.get_block_type_tensors(BlockType::C4);
  ASSERT_TRUE(c4_tensors.count(KVCacheTensorRole::KEY) > 0);
  ASSERT_TRUE(c4_tensors.count(KVCacheTensorRole::INDEX) > 0);
  EXPECT_EQ(c4_tensors.at(KVCacheTensorRole::KEY).size(0),
            scale_host_block_count(kC4Count, kHostFactor));
  EXPECT_EQ(c4_tensors.at(KVCacheTensorRole::INDEX).scalar_type(),
            get_dsv4_cache_policy(options.dtype()).index_dtype);

  // C128 host cache: key only (no index).
  KVCache c128_host(shape, options, BlockType::C128, /*layer_count=*/1);
  const BlockTypeTensorMap c128_tensors =
      c128_host.get_block_type_tensors(BlockType::C128);
  ASSERT_TRUE(c128_tensors.count(KVCacheTensorRole::KEY) > 0);
  EXPECT_TRUE(c128_tensors.count(KVCacheTensorRole::INDEX) == 0);
  EXPECT_EQ(c128_tensors.at(KVCacheTensorRole::KEY).size(0),
            scale_host_block_count(kC128Count, kHostFactor));
}

}  // namespace xllm
