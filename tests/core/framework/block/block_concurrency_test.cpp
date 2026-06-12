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

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "framework/block/block_manager_impl.h"

namespace xllm {

// Block aliases of one physical block are copied and dropped from many
// threads (hierarchy offload callbacks, disagg PD threadpool); the shared
// refcount must neither lose updates nor free the block prematurely.
TEST(BlockConcurrencyTest, RefCountStaysConsistentAcrossThreads) {
  BlockManager::Options options;
  options.num_blocks(2).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(blocks.size(), 1u);
  const Block& shared_block = blocks[0];
  EXPECT_EQ(shared_block.ref_count(), 1u);

  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 20000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));

  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([&shared_block, &start, kNumIterations]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        Block copy = shared_block;
        Block another = copy;
        Block moved = std::move(copy);
        another = moved;
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(shared_block.ref_count(), 1u);
  EXPECT_EQ(manager.num_free_blocks(), 0u);

  blocks.clear();
  EXPECT_EQ(manager.num_free_blocks(), 1u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

// The plain BlockManagerImpl (no concurrent wrapper) must keep its free list
// consistent under parallel allocate/deallocate: free() is re-entered from
// Block destructors on whatever thread drops the last alias.
TEST(BlockConcurrencyTest, PlainAllocatorAllocatesAndFreesConcurrently) {
  BlockManager::Options options;
  options.num_blocks(65).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 10000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));

  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([&manager, &start, kNumIterations, i]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      const size_t batch = static_cast<size_t>(i % 2) + 1;
      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        std::vector<Block> blocks = manager.allocate(batch);
        if (blocks.empty()) {
          std::this_thread::yield();
          continue;
        }

        manager.deallocate(blocks);
        blocks.clear();
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

// Blocks allocated on one thread and dropped on another (the hierarchy
// offload pattern) must return to the free list exactly once.
TEST(BlockConcurrencyTest, CrossThreadHandoffFreesExactlyOnce) {
  BlockManager::Options options;
  options.num_blocks(33).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  constexpr int32_t kNumRounds = 2000;
  constexpr size_t kBatch = 4;
  std::mutex handoff_mutex;
  std::vector<Block> handoff;
  std::atomic<bool> done{false};

  std::thread consumer([&handoff_mutex, &handoff, &done]() {
    while (true) {
      std::vector<Block> taken;
      {
        std::lock_guard<std::mutex> lock(handoff_mutex);
        taken.swap(handoff);
      }
      const bool exit_after_drain = done.load(std::memory_order_acquire);
      taken.clear();
      if (exit_after_drain) {
        std::lock_guard<std::mutex> lock(handoff_mutex);
        if (handoff.empty()) {
          return;
        }
      }
      std::this_thread::yield();
    }
  });

  for (int32_t round = 0; round < kNumRounds; ++round) {
    std::vector<Block> blocks = manager.allocate(kBatch);
    if (blocks.empty()) {
      std::this_thread::yield();
      continue;
    }
    std::lock_guard<std::mutex> lock(handoff_mutex);
    for (Block& block : blocks) {
      handoff.emplace_back(std::move(block));
    }
  }

  done.store(true, std::memory_order_release);
  consumer.join();

  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

}  // namespace xllm
