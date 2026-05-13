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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode) {
  check_tensor(x, "x", "causal_conv1d");
  check_tensor(weight, "weight", "causal_conv1d");
  check_tensor(conv_state, "conv_state", "causal_conv1d");

  c10::optional<torch::Tensor> bias_tensor = c10::nullopt;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    bias_tensor = bias_opt.value();
  }

  torch::Tensor output = torch::empty(x.sizes(), x.options());
  EXEC_NPU_CMD(aclnnCausalConv1d,
               x,
               weight,
               bias_tensor,
               conv_state,
               query_start_loc_opt,
               cache_indices_opt,
               initial_state_mode_opt,
               num_accepted_tokens_opt,
               activation_mode,
               pad_slot_id,
               run_mode,
               output);
  return output;
}

}  // namespace xllm::kernel::npu
