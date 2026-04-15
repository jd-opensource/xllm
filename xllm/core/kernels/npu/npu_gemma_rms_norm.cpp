/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "npu_ops_api.h"

// Include ascendc_ops_api.h for npu_ops::npu_gemma_rms_norm
#include "ascendc_npu/ascendc_ops_api.h"

namespace xllm::kernel::npu {

void npu_gemma_rms_norm(const torch::Tensor& x,
                        const torch::Tensor& gamma,
                        double epsilon,
                        torch::Tensor& rstdOut,
                        torch::Tensor& yOut) {
  npu_ops::npu_gemma_rms_norm(x, gamma, epsilon, rstdOut, yOut);
}

}  // namespace xllm::kernel::npu
