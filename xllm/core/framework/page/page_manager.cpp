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

#include "page_manager.h"

#include "common/global_flags.h"
#include "multi_layer_xtensor_transfer.h"
#include "util/type_traits.h"

namespace xllm {

PageManager::PageManager(const page::Options& options,
                         const torch::Device& device)
    : options_(options), device_(device) {
  page_allocator_ = std::make_unique<PageAllocator>(options_, device_);
  add_multi_layer_kv_xtensors();
}

void PageManager::add_multi_layer_kv_xtensors() {
  MultiLayerXTensorPair multi_layer_kv_xtensor =
      MultiLayerXTensorTransfer::get_instance().move_multi_layer_xtensor(
          device_.index());

  multi_layer_kv_xtensor_ =
      std::make_pair(std::move(multi_layer_kv_xtensor.first),
                     std::move(multi_layer_kv_xtensor.second));
}

int32_t PageManager::allocate_seq_id() {
  int32_t seq_id = -1;
  // TODO: refine this
  seq_id = multi_layer_kv_xtensor_.first->allocate_seq_id();
  seq_id = multi_layer_kv_xtensor_.second->allocate_seq_id();
}

void PageManager::deallocate_seq_id(int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "seq_id is not valid!";

  multi_layer_kv_xtensor_.first->deallocate_seq_id(seq_id);
  multi_layer_kv_xtensor_.second->deallocate_seq_id(seq_id);
}

// num_tokens is the number of all tokens in sequence
bool PageManager::allocate(int32_t& seq_id, size_t num_tokens) {
  if (seq_id < 0) {
    seq_id = allocate_seq_id();
  }

  const size_t k_num_pages =
      multi_layer_kv_xtensor_.first->get_num_pages_per_layer(seq_id);
  const size_t v_num_pages =
      multi_layer_kv_xtensor_.second->get_num_pages_per_layer(seq_id);
  const int64_t cache_size_per_token = options_.cache_size_per_token();

  const size_t k_num_pages_needed =
      (num_tokens * cache_size_per_token + FLAGS_granularity_size - 1) /
      FLAGS_granularity_size;
  size_t v_num_pages_needed = k_num_pages_needed;
  if (FLAGS_enable_mla) {
    v_num_pages_needed =
        (num_tokens * cache_size_per_token / 8 + FLAGS_granularity_size - 1) /
        FLAGS_granularity_size;
  }

  if ((k_num_pages_needed <= k_num_pages) &&
      (v_num_pages_needed <= v_num_pages)) {
    return true;
  }

  const size_t k_num_additional_pages = k_num_pages_needed - k_num_pages;
  const size_t v_num_additional_pages = v_num_pages_needed - v_num_pages;

  if (!has_enough_pages(k_num_additional_pages, v_num_additional_pages)) {
    return false;
  }

  if (k_num_additional_pages > 0) {
    auto& multi_layer_k_xtensor = multi_layer_kv_xtensor_.first;
    std::vector<uint32_t> new_k_phy_page_ids =
        page_allocator_->allocate(k_num_additional_pages);
    multi_layer_k_xtensor->append_phy_pages(seq_id, new_k_phy_page_ids);

    std::vector<uint32_t> k_phy_page_ids =
        multi_layer_k_xtensor->get_phy_page_ids(seq_id);
    for (int64_t layer_idx = 0; layer_idx < options_.num_layers();
         ++layer_idx) {
      VirPtr k_vit_ptr = multi_layer_k_xtensor->get_vir_ptr(seq_id, layer_idx);
      page_allocator_->batch_map(
          k_vit_ptr, k_phy_page_ids, k_num_additional_pages, layer_idx);
    }

    num_used_pages_per_layer_ += k_num_additional_pages;
  }

  if (v_num_additional_pages > 0) {
    auto& multi_layer_v_xtensor = multi_layer_kv_xtensor_.second;
    std::vector<uint32_t> new_v_phy_page_ids =
        page_allocator_->allocate(v_num_additional_pages);
    multi_layer_v_xtensor->append_phy_pages(seq_id, new_v_phy_page_ids);

    std::vector<uint32_t> v_phy_page_ids =
        multi_layer_v_xtensor->get_phy_page_ids(seq_id);
    for (int64_t layer_idx = 0; layer_idx < options_.num_layers();
         ++layer_idx) {
      VirPtr v_vit_ptr = multi_layer_v_xtensor->get_vir_ptr(seq_id, layer_idx);
      page_allocator_->batch_map(
          v_vit_ptr, v_phy_page_ids, v_num_additional_pages, layer_idx);
    }

    num_used_pages_per_layer_ += v_num_additional_pages;
  }

  return true;
}

void PageManager::deallocate(int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "seq_id is not valid!";

  const size_t k_num_pages_used_per_layer =
      multi_layer_kv_xtensor_.first->get_num_pages_per_layer(seq_id);
  const size_t v_num_pages_used_per_layer =
      multi_layer_kv_xtensor_.second->get_num_pages_per_layer(seq_id);

  auto& multi_layer_k_xtensor = multi_layer_kv_xtensor_.first;
  multi_layer_k_xtensor->free(seq_id);
  std::vector<uint32_t> k_used_page_ids =
      multi_layer_k_xtensor->get_phy_page_ids(seq_id);
  page_allocator_->deallocate(k_used_page_ids);

  auto& multi_layer_v_xtensor = multi_layer_kv_xtensor_.second;
  multi_layer_v_xtensor->free(seq_id);
  std::vector<uint32_t> v_used_page_ids =
      multi_layer_v_xtensor->get_phy_page_ids(seq_id);
  page_allocator_->deallocate(v_used_page_ids);

  // deallocate seq id for sequence
  deallocate_seq_id(seq_id);

  // key cache
  num_used_pages_per_layer_ = num_used_pages_per_layer_ -
                              k_num_pages_used_per_layer -
                              v_num_pages_used_per_layer;
}

void PageManager::cache(int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "seq_id is not valid!";

  const size_t k_num_pages_used_per_layer =
      multi_layer_kv_xtensor_.first->get_num_pages_per_layer(seq_id);
  const size_t v_num_pages_used_per_layer =
      multi_layer_kv_xtensor_.second->get_num_pages_per_layer(seq_id);

  // key cache
  num_used_pages_per_layer_ = num_used_pages_per_layer_ -
                              k_num_pages_used_per_layer -
                              v_num_pages_used_per_layer;
}

folly::SemiFuture<bool> PageManager::allocate_async(int32_t& seq_id,
                                                    size_t num_tokens) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, seq_id, num_tokens, promise = std::move(promise)]() mutable {
        const bool success = this->allocate(seq_id, num_tokens);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<folly::Unit> PageManager::deallocate_async(int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    this->deallocate(seq_id);
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<folly::Unit> PageManager::cache_async(int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    this->cache(seq_id);
    promise.setValue();
  });
  return future;
}

size_t PageManager::num_free_pages_per_layer() const {
  return page_allocator_->get_num_free_phy_pages_per_layer();
}

size_t PageManager::num_used_pages_per_layer() const {
  return num_used_pages_per_layer_;
}

double PageManager::kv_cache_utilization() const {
  return static_cast<double>(num_used_pages_per_layer_) /
         page_allocator_->get_num_total_phy_pages_per_layer();
}

folly::SemiFuture<size_t> PageManager::num_free_pages_per_layer_async() {
  folly::Promise<size_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const size_t num_free_pages_per_layer = this->num_free_pages_per_layer();
    promise.setValue(num_free_pages_per_layer);
  });
  return future;
}

folly::SemiFuture<size_t> PageManager::num_used_pages_per_layer_async() {
  folly::Promise<size_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const size_t num_used_pages_per_layer = this->num_used_pages_per_layer();
    promise.setValue(num_used_pages_per_layer);
  });
  return future;
}

bool PageManager::has_enough_pages(size_t k_num_pages_needed,
                                   size_t v_num_pages_needed) {
  // still have enough pages
  if (k_num_pages_needed + v_num_pages_needed <=
      page_allocator_->get_num_free_phy_pages_per_layer()) {
    return true;
  }
  return false;
}

}  // namespace xllm
