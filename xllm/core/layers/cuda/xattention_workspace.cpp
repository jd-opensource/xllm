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

#include "xattention_workspace.h"

#include <glog/logging.h>

namespace xllm::layer::xattention {

void XAttentionWorkspace::initialize(const torch::Device& device,
                                     int64_t int_workspace_size) {
  CHECK_GT(int_workspace_size, 0)
      << "int_workspace_size must be greater than 0, got "
      << int_workspace_size;

  const bool is_already_initialized =
      two_stage_unshared_int_workspace_buffer_.defined() &&
      two_stage_unshared_page_locked_int_workspace_buffer_.defined() &&
      two_stage_unshared_int_workspace_buffer_.device() == device &&
      two_stage_unshared_int_workspace_buffer_.size(0) == int_workspace_size;
  if (is_already_initialized) {
    return;
  }

  LOG(INFO) << "XAttentionWorkspace initialize on device: " << device
            << ", int_workspace_size: " << int_workspace_size;

  two_stage_unshared_int_workspace_buffer_ = torch::empty(
      {int_workspace_size}, torch::dtype(torch::kUInt8).device(device));
  two_stage_unshared_page_locked_int_workspace_buffer_ = torch::empty(
      {int_workspace_size},
      torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
}

torch::Tensor
XAttentionWorkspace::get_two_stage_unshared_int_workspace_buffer() {
  return two_stage_unshared_int_workspace_buffer_;
}

torch::Tensor
XAttentionWorkspace::get_two_stage_unshared_page_locked_int_workspace_buffer() {
  return two_stage_unshared_page_locked_int_workspace_buffer_;
}

}  // namespace xllm::layer::xattention
