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

#pragma once

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "multi_layer_xtensor_transfer.h"
#include "options.h"
#include "page_allocator.h"
#include "util/threadpool.h"

namespace xllm {

class PageManager {
 public:
  explicit PageManager(const page::Options& options,
                       const torch::Device& device);

  ~PageManager() = default;

  bool allocate(int32_t& seq_id, size_t num_tokens);
  void deallocate(int32_t seq_id);
  void cache(int32_t seq_id);

  folly::SemiFuture<bool> allocate_async(int32_t& seq_id, size_t num_tokens);
  folly::SemiFuture<folly::Unit> deallocate_async(int32_t seq_id);
  folly::SemiFuture<folly::Unit> cache_async(int32_t seq_id);

  size_t num_free_pages_per_layer() const;
  size_t num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

  folly::SemiFuture<size_t> num_free_pages_per_layer_async();
  folly::SemiFuture<size_t> num_used_pages_per_layer_async();

 private:
  void add_multi_layer_kv_xtensors();
  // allocate seq id for sequence
  int32_t allocate_seq_id();
  // release seq id for sequence
  void deallocate_seq_id(int32_t seq_id);
  bool has_enough_pages(size_t k_num_pages_needed, size_t v_num_pages_needed);

 private:
  page::Options options_;
  torch::Device device_;
  std::unique_ptr<PageAllocator> page_allocator_;
  MultiLayerXTensorPair multi_layer_kv_xtensor_;
  size_t num_used_pages_per_layer_ = 0;
  ThreadPool threadpool_;
};

}  // namespace xllm