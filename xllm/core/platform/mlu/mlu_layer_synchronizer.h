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

#include <c10/core/Event.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace xllm {

class MLULayerSynchronizerImpl final {
 public:
  explicit MLULayerSynchronizerImpl(int64_t num_layers);
  ~MLULayerSynchronizerImpl() = default;

  bool synchronize_layer(int64_t layer_index);
  bool record_current(int64_t layer_index, int32_t device_index);
  uint32_t get_event_size() const { return events_.size(); }

 private:
  std::vector<c10::Event> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
};

}  // namespace xllm
