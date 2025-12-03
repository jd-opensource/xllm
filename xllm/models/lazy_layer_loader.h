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

#include <acl/acl.h>
#include <torch_npu/torch_npu.h>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "core/platform/stream.h"
#include "core/util/threadpool.h"

namespace xllm {

/**
 * LazyLayerLoader - Elegant abstraction for on-demand layer weight loading
 *
 * Design principles:
 * - Defers layer weights loading until first forward pass
 * - Sequential layer loading on dedicated stream
 * - Per-layer ACL events for fine-grained synchronization
 */

class LazyLayerLoader {
 public:
  /**
   * Callback to load, verify, and merge a single layer
   * @param layer_idx Index of the layer to process
   */
  using LayerLoader = std::function<void(int32_t layer_idx)>;

  /**
   * Constructor
   * @param num_layers Total number of layers in the model
   * @param device_id NPU device ID
   */
  LazyLayerLoader(int32_t num_layers, int32_t device_id);

  ~LazyLayerLoader();

  LazyLayerLoader(const LazyLayerLoader&) = delete;
  LazyLayerLoader& operator=(const LazyLayerLoader&) = delete;
  LazyLayerLoader(LazyLayerLoader&&) = delete;
  LazyLayerLoader& operator=(LazyLayerLoader&&) = delete;

  /**
   * reset all events to unrecorded state.
   */
  void reset_events();
  /**
   * Start asynchronous loading of all layers
   * @param handle Callback that loads/verifies/merges a layer
   */
  void start_async_loading(LayerLoader handle);

  /**
   * Wait until specified layer is fully loaded and ready
   * @param layer_idx Index of the layer to wait for
   */
  void wait_for_layer(int32_t layer_idx);

 private:
  void load_layer(int32_t layer_idx, LayerLoader processor);

  const int32_t num_layers_;
  const int32_t device_id_;

  c10_npu::NPUStream load_stream_;
  std::unique_ptr<ThreadPool> threadpool_;
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_recorded_;
};

}  // namespace xllm
