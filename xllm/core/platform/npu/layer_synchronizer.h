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

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace xllm {

class LayerSynchronizer {
 public:
  LayerSynchronizer() : timeout_duration_(std::chrono::milliseconds(0)) {}
  LayerSynchronizer(const int64_t layer_num, const int32_t timeout = 0);
  virtual ~LayerSynchronizer();

  void notify(const int64_t layer_index);

  virtual bool synchronize_layer(const int64_t layer_index);
  virtual uint32_t get_event_size() { return completed_.size(); };

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<bool> completed_;
  const std::chrono::milliseconds timeout_duration_;
};

}  // namespace xllm
