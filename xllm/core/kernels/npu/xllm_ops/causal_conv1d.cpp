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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_causal_conv1d.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

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

  aclTensor* x_ids = nullptr;
  aclTensor* weight_ids = nullptr;
  aclTensor* bias_ids = nullptr;
  aclTensor* conv_state_ids = nullptr;
  aclTensor* output_ids = nullptr;
  aclIntArray* query_start_loc_ids = nullptr;
  aclIntArray* cache_indices_ids = nullptr;
  aclIntArray* initial_state_mode_ids = nullptr;
  aclIntArray* num_accepted_tokens_ids = nullptr;

  int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  create_acltensor(&x_ids, x);
  create_acltensor(&weight_ids, weight);
  create_acltensor(&conv_state_ids, conv_state);
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    create_acltensor(&bias_ids, bias_opt.value());
  }
  query_start_loc_ids =
      aclCreateIntArray(query_start_loc_opt.data(), query_start_loc_opt.size());
  cache_indices_ids =
      aclCreateIntArray(cache_indices_opt.data(), cache_indices_opt.size());
  initial_state_mode_ids = aclCreateIntArray(initial_state_mode_opt.data(),
                                             initial_state_mode_opt.size());
  num_accepted_tokens_ids = aclCreateIntArray(num_accepted_tokens_opt.data(),
                                              num_accepted_tokens_opt.size());

  torch::Tensor output = torch::empty(x.sizes(), x.options());
  create_acltensor(&output_ids, output);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  CHECK_ACL_SUCCESS(aclnnCausalConv1dGetWorkspaceSize(x_ids,
                                                      weight_ids,
                                                      bias_ids,
                                                      conv_state_ids,
                                                      query_start_loc_ids,
                                                      cache_indices_ids,
                                                      initial_state_mode_ids,
                                                      num_accepted_tokens_ids,
                                                      activation_mode,
                                                      pad_slot_id,
                                                      run_mode,
                                                      output_ids,
                                                      &workspace_size,
                                                      &executor),
                    "causal_conv1d: failed to get workspace size");

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "causal_conv1d: failed to allocate workspace");
  }

  CHECK_ACL_SUCCESS(
      aclnnCausalConv1d(workspace_addr, workspace_size, executor, stream),
      "causal_conv1d: failed to perform causal conv1d");

  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "causal_conv1d: failed to synchronize stream");

  aclDestroyTensor(x_ids);
  aclDestroyTensor(weight_ids);
  aclDestroyTensor(conv_state_ids);
  aclDestroyTensor(output_ids);
  if (bias_ids != nullptr) {
    aclDestroyTensor(bias_ids);
  }
  aclDestroyIntArray(query_start_loc_ids);
  aclDestroyIntArray(cache_indices_ids);
  aclDestroyIntArray(initial_state_mode_ids);
  aclDestroyIntArray(num_accepted_tokens_ids);

  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "causal_conv1d: failed to free workspace");
  }

  return output;
}
}  // namespace xllm::kernel::npu