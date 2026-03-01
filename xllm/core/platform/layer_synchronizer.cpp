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

#include "layer_synchronizer.h"

#include <glog/logging.h>

namespace xllm {

LayerSynchronizer::LayerSynchronizer(const int64_t layer_num,
                                     const int32_t timeout)
    : completed_(layer_num, false),
      timeout_duration_(timeout > 0 ? std::chrono::milliseconds(timeout)
                                    : std::chrono::milliseconds(0)) {}

LayerSynchronizer::~LayerSynchronizer() = default;

void LayerSynchronizer::notify(const int64_t layer_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(layer_index, static_cast<int64_t>(completed_.size()))
      << "Index out of range: " << layer_index
      << ", size: " << completed_.size();

  completed_[layer_index] = true;
  cv_.notify_all();
}

bool LayerSynchronizer::synchronize_layer(const int64_t layer_index) {
  std::unique_lock<std::mutex> lock(mutex_);

  CHECK_LT(layer_index, static_cast<int64_t>(completed_.size()))
      << "Index out of range: " << layer_index
      << ", size: " << completed_.size();

  if (timeout_duration_.count() == 0) {
    cv_.wait(lock, [this, layer_index] { return completed_[layer_index]; });
    return true;
  }

  bool result = cv_.wait_for(lock, timeout_duration_, [this, layer_index] {
    return completed_[layer_index];
  });

  if (!result) {
    LOG(WARNING) << "Wait for " << layer_index << " timeout after "
                 << timeout_duration_.count() << " ms";
  }

  return result;
}

}  // namespace xllm
