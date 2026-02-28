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

#include "core/common/macros.h"

namespace xllm::layer::xattention {

class XAttentionWorkspace {
 public:
  static XAttentionWorkspace& get_instance() {
    static XAttentionWorkspace instance;
    return instance;
  };

  void initialize(const torch::Device& device, int64_t int_workspace_size);

  torch::Tensor get_two_stage_unshared_int_workspace_buffer();

  torch::Tensor get_two_stage_unshared_page_locked_int_workspace_buffer();

 private:
  XAttentionWorkspace() = default;
  ~XAttentionWorkspace() = default;
  DISALLOW_COPY_AND_ASSIGN(XAttentionWorkspace);

  torch::Tensor two_stage_unshared_int_workspace_buffer_;
  torch::Tensor two_stage_unshared_page_locked_int_workspace_buffer_;
};

}  // namespace xllm::layer::xattention
