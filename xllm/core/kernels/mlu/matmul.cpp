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

#include "torch_mlu/csrc/aten/cnnl/cnnlHandle.h"
#include "torch_mlu/csrc/framework/core/mlu_guard.h"
#include "torch_mlu_ops.h"
#include "torch_ops_api.h"

namespace xllm::mlu {

at::Tensor matmul(const at::Tensor& a,
                  const at::Tensor& b,
                  const c10::optional<at::Tensor>& bias,
                  const c10::optional<at::Tensor>& c,
                  double alpha,
                  double beta) {
  return tmo::torch_api::matmul(a,
                                b,
                                bias,
                                c,
                                c10::nullopt,
                                c10::nullopt,
                                c10::nullopt,
                                c10::nullopt,
                                "none",
                                alpha,
                                beta,
                                true,
                                true,
                                1.0,
                                1.0,
                                false,
                                true);
}

}  // namespace xllm::mlu
