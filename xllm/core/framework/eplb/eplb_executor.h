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

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "platform/stream_helper.h"
#include "runtime/forward_params.h"

namespace xllm {

class EplbExecutor final {
 public:
  using Callback = std::function<void(int32_t)>;
  EplbExecutor(CausalLM* model);

  virtual ~EplbExecutor();

  // Reset the ready layer ID marker to -1 (no layer ready)
  void reset_ready_layer_id();

  // Get the currently ready layer ID
  // return int32_t Layer ID with prepared weights (-1 if none)
  int32_t get_ready_layer_id() const;

  // Execute EPLB operation based on coordination info
  // param eplb_info Contains layer preparation/activation instructions
  void eplb_execute(const EplbInfo& eplb_info);

 private:
  struct Task {
    int32_t layer_id;
    std::vector<int32_t> expert_ids;
    Callback callback;
  };

  void eplb_worker_loop();
  void prepare_expert_weight_async(int32_t layer_id,
                                   const std::vector<int32_t>& expert_ids,
                                   Callback callback = nullptr);
  CausalLM* model_;
  std::thread eplb_worker_;
  std::queue<Task> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;

  mutable std::mutex ready_mutex_;
  int32_t ready_layer_id_ = -1;
  std::unique_ptr<StreamHelper> eplb_stream_helper_;
};

}  // namespace xllm
