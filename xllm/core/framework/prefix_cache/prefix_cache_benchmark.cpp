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

#include <benchmark/benchmark.h>
#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "prefix_cache_hash.h"

using namespace xllm;

static void BM_HashSearch(benchmark::State& state) {
  const uint32_t block_size = 16;
  const uint32_t total_blocks = state.range(0);
  const uint32_t token_id_count = state.range(1);

  // LOG(INFO) << "total blocks " << total_blocks << ", token_id_count " <<
  // token_id_count;

  assert((token_id_count / block_size) < total_blocks);

  state.PauseTiming();
  BlockManager::Options options;
  options.num_blocks(n_blocks + 1).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCacheHash prefix_cache(block_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned> dist(0, 65535);

  std::vector<int32_t> token_ids(token_id_count);
  std::generate(
      token_ids.begin(), token_ids.end(), [&]() { return dist(gen); });

  uint32_t n_blocks = token_id_count / block_size;

  std::vector<Block> token_blocks = block_manager.allocate(n_blocks);
  Slice<Block> slice_token_blocks(token_blocks);
  Slice<int32_t> slice_token_ids(token_ids);
  std::vector<int32_t> match_token_ids(token_ids);

  prefix_cache.insert(slice_token_ids, slice_token_blocks);
  state.ResumeTiming();

  size_t count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(prefix_cache.match(match_token_ids));

    ++count;
  }

  state.counters["iter count"] = count;
}

BENCHMARK(BM_HashSearch)
    ->Args({2048, 5000})
    ->Unit(benchmark::TimeUnit::kMillisecond)
    ->UseRealTime()
    ->Iterations(100)
    ->Repetitions(20)
    ->ReportAggregatesOnly(true);
