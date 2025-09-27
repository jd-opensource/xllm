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

#include "xtensor_manager_client.h"

namespace xllm {

bool XTensorManagerClient::allocate(int32_t& seq_id, size_t num_tokens) {
  return xtensor_manager_->allocate(seq_id, num_tokens);
}

void XTensorManagerClient::deallocate(int32_t seq_id) {
  xtensor_manager_->deallocate(seq_id);
}

folly::SemiFuture<bool> XTensorManagerClient::allocate_async(
    int32_t& seq_id,
    size_t num_tokens) {
  return xtensor_manager_->allocate_async(seq_id, num_tokens);
}

folly::SemiFuture<folly::Unit> XTensorManagerClient::deallocate_async(
    int32_t seq_id) {
  return xtensor_manager_->deallocate_async(seq_id);
}

size_t XTensorManagerClient::num_free_pages_per_layer() const {
  return xtensor_manager_->num_free_pages_per_layer();
}

size_t XTensorManagerClient::num_used_pages_per_layer() const {
  return xtensor_manager_->num_used_pages_per_layer();
}

double XTensorManagerClient::kv_cache_utilization() const {
  return xtensor_manager_->kv_cache_utilization();
}

folly::SemiFuture<size_t>
XTensorManagerClient::num_free_pages_per_layer_async() {
  return xtensor_manager_->num_free_pages_per_layer_async();
}

folly::SemiFuture<size_t>
XTensorManagerClient::num_used_pages_per_layer_async() {
  return xtensor_manager_->num_used_pages_per_layer_async();
}

}  // namespace xllm