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
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_select_unshared_kv.h"
#include "acltensor_utils.h"
#include "util/tensor_helper.h"

namespace xllm_ops {
void cache_select(const torch::Tensor& beam_index,
                  std::vector<torch::Tensor> x_key_block,
                  std::vector<torch::Tensor> x_value_block,
                  const torch::Tensor& block_table,
                  const torch::Tensor& group_offset,
                  int64_t decode_step,
                  int64_t beam_size,
                  int64_t layer_num);
}  // namespace xllm_ops