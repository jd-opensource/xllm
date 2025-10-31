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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

void rmsnorm(torch::Tensor& output,
             torch::Tensor& input,
             torch::Tensor& weight,
             double eps) {
  get_module("norm")->GetFunction("rmsnorm").value()(to_ffi_tensor(output),
                                                     to_ffi_tensor(input),
                                                     to_ffi_tensor(weight),
                                                     eps,
                                                     support_pdl());
}

}  // namespace xllm::kernel::cuda