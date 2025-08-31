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

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "util/tensor_helper.h"

namespace xllm_ops_utils {
struct type_info {
  static aclDataType get_acl_type(const torch::ScalarType& dtype);
};

void create_acltensor(aclTensor** tensor, const torch::Tensor& tensor_data);
void check_tensor(const torch::Tensor& t,
                  const std::string& name,
                  const std::string& func_name = "");
void check_tensor_shapes_equal(const torch::Tensor& a,
                               const torch::Tensor& b,
                               const std::string& func_name = "");
}  // namespace xllm_ops_utils