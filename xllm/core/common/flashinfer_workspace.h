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

#include <torch/torch.h>

#include <cstdint>

#include "macros.h"

namespace xllm {

class FlashinferWorkspace {
 public:
  static FlashinferWorkspace& get_instance() {
    static FlashinferWorkspace instance;
    return instance;
  };

  void initialize(const torch::Device& device);

  torch::Tensor get_float_workspace_buffer();
  torch::Tensor get_int_workspace_buffer();
  torch::Tensor get_page_locked_int_workspace_buffer();

 private:
  FlashinferWorkspace() = default;
  ~FlashinferWorkspace() = default;
  DISALLOW_COPY_AND_ASSIGN(FlashinferWorkspace);

  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;
};

}  // namespace xllm