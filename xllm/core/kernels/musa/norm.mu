/* Copyright 2025 The vLLM Authors and The xLLM Authors. All Rights Reserved.

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

#include "musa_ops_api.h"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu


namespace xllm::kernel::musa {

void rms_norm(torch::Tensor output,  // [..., hidden_size]
              torch::Tensor input,   // [..., hidden_size]
              torch::Tensor weight,  // [hidden_size]
              double eps) {
 
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
 
}

}  // namespace xllm::kernel::musa
