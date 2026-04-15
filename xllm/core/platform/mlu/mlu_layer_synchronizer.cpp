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

#include "platform/mlu/mlu_layer_synchronizer.h"

#include <framework/core/MLUStream.h>

#include <thread>

namespace xllm {

MLULayerSynchronizerImpl::MLULayerSynchronizerImpl(int64_t num_layers)
    : events_(), event_record_flags_(static_cast<size_t>(num_layers)) {
  events_.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    events_.emplace_back(c10::DeviceType::PrivateUse1);
  }
}

bool MLULayerSynchronizerImpl::synchronize_layer(int64_t layer_index) {
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  events_[layer_index].synchronize();
  return true;
}

bool MLULayerSynchronizerImpl::record_current(int64_t layer_index,
                                              int32_t device_index) {
  c10::Stream current_stream =
      torch_mlu::getCurrentMLUStream(device_index).unwrap();
  events_[layer_index].record(current_stream);
  event_record_flags_[layer_index].store(true, std::memory_order_release);
  return true;
}

}  // namespace xllm
