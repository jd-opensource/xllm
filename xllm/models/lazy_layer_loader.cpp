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

#include "lazy_layer_loader.h"

#include <glog/logging.h>

namespace xllm {

LazyLayerLoader::LazyLayerLoader(int32_t num_layers, int32_t device_id)
    : num_layers_(num_layers),
      device_id_(device_id),
      load_stream_(c10_npu::getNPUStreamFromPool()),
      events_(num_layers, nullptr),
      event_recorded_(num_layers) {
  uint32_t flags = ACL_EVENT_SYNC;
  threadpool_ = std::make_unique<ThreadPool>(1);

  for (int32_t i = 0; i < num_layers; ++i) {
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK_EQ(ret, ACL_SUCCESS) << "Failed to create event for layer " << i;
    event_recorded_[i].store(false, std::memory_order_relaxed);
  }

  LOG(INFO) << "lazy layer loader initialized for " << num_layers << " layers";
}

LazyLayerLoader::~LazyLayerLoader() {
  for (int i = 0; i < events_.size(); i++) {
    if (events_[i] != nullptr) {
      aclrtDestroyEvent(events_[i]);
    }
  }
}

void LazyLayerLoader::start_async_loading(LayerLoader handle) {
  LOG(INFO) << "starting asynchronous layer loading for " << num_layers_
            << " layers";
  threadpool_->schedule([this, handle]() {
    for (int32_t i = 0; i < num_layers_; ++i) {
      load_layer(i, handle);
    }
  });
}

void LazyLayerLoader::load_layer(int32_t layer_idx, LayerLoader handle) {
  c10_npu::SetDevice(device_id_);
  auto stream_guard = c10::StreamGuard(load_stream_.unwrap());
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  handle(layer_idx);

  // Capture current layer load task.
  auto ret = aclrtRecordEvent(events_[layer_idx], load_stream_.stream());
  CHECK_EQ(ret, ACL_SUCCESS)
      << "failed to record event for layer " << layer_idx;

  event_recorded_[layer_idx].store(true, std::memory_order_release);
}

void LazyLayerLoader::wait_for_layer(int32_t layer_idx) {
  while (!event_recorded_[layer_idx].load(std::memory_order_acquire)) {
    // busy wait.
  }

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  auto ret = aclrtStreamWaitEvent(stream, events_[layer_idx]);
  CHECK_EQ(ret, ACL_SUCCESS) << "failed to sync layer " << layer_idx;
  ret = aclrtResetEvent(events_[layer_idx], stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "failed to reset event " << layer_idx;
}

void LazyLayerLoader::reset_events() {
  for (int32_t i = 0; i < num_layers_; ++i) {
    event_recorded_[i].store(false, std::memory_order_relaxed);
  }
}
}  // namespace xllm
