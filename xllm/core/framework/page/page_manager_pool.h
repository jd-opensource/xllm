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
#include <memory>
#include <vector>

#include "common/macros.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "page_manager_client.h"
#include "page_manager_server.h"

namespace xllm {
class PageManagerPool {
 public:
  explicit PageManagerPool(const page::Options& options, int32_t dp_size);
  ~PageManagerPool() = default;

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void cache(Sequence* sequence);

  std::vector<size_t> num_free_pages_per_layer() const;
  std::vector<size_t> num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerPool);
  void setup_single_node_page_managers();
  void setup_multi_node_page_managers(const std::string& master_node_addr);
  int32_t get_manager_with_max_free_pages() const;
  int32_t get_dp_rank(Sequence* sequence) const;

 private:
  page::Options options_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  std::vector<std::shared_ptr<PageManagerClient>> page_manager_clients_;
  std::vector<std::shared_ptr<PageManager>> page_managers_;
  std::vector<std::unique_ptr<PageManagerServer>> page_manager_servers_;
};
}  // namespace xllm