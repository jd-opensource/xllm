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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int start_expert_id,
            int expert_size) {
  tmo::torch_api::active(input,
                         output,
                         bias,
                         cusum_token_count,
                         act_mode,
                         is_gated,
                         start_expert_id,
                         expert_size);
}
}  // namespace xllm::kernel::mlu