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
#include "aclnn_recurrent_gated_delta_rule.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "npu_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor npu_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk) {
  check_tensor(query, "query", "recurrent_gated_delta_rule");
  check_tensor(key, "key", "recurrent_gated_delta_rule");
  check_tensor(value, "value", "recurrent_gated_delta_rule");
  check_tensor(state, "state", "recurrent_gated_delta_rule");

  aclTensor* query_ids = nullptr;
  aclTensor* key_ids = nullptr;
  aclTensor* value_ids = nullptr;
  aclTensor* state_ids = nullptr;
  aclTensor* beta_ids = nullptr;
  aclTensor* actual_seq_lengths_ids = nullptr;
  aclTensor* ssm_state_indices_ids = nullptr;
  aclTensor* num_accepted_tokens_ids = nullptr;
  aclTensor* g_ids = nullptr;
  aclTensor* gk_ids = nullptr;
  aclTensor* out_ids = nullptr;

  int32_t device_id = query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  create_acltensor(&query_ids, query);
  create_acltensor(&key_ids, key);
  create_acltensor(&value_ids, value);
  create_acltensor(&state_ids, state);

  if (beta.has_value() && beta.value().defined()) {
    create_acltensor(&beta_ids, beta.value());
  }
  if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
    create_acltensor(&actual_seq_lengths_ids, actual_seq_lengths.value());
  }
  if (ssm_state_indices.has_value() && ssm_state_indices.value().defined()) {
    create_acltensor(&ssm_state_indices_ids, ssm_state_indices.value());
  }
  if (num_accepted_tokens.has_value() &&
      num_accepted_tokens.value().defined()) {
    create_acltensor(&num_accepted_tokens_ids, num_accepted_tokens.value());
  }
  if (g.has_value() && g.value().defined()) {
    create_acltensor(&g_ids, g.value());
  }
  if (gk.has_value() && gk.value().defined()) {
    create_acltensor(&gk_ids, gk.value());
  }

  at::Tensor out_result = at::empty_like(value);
  create_acltensor(&out_ids, out_result);

  float scale_value = static_cast<float>(scale.value());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  CHECK_ACL_SUCCESS(
      aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(query_ids,
                                                   key_ids,
                                                   value_ids,
                                                   beta_ids,
                                                   state_ids,
                                                   actual_seq_lengths_ids,
                                                   ssm_state_indices_ids,
                                                   g_ids,
                                                   gk_ids,
                                                   num_accepted_tokens_ids,
                                                   scale_value,
                                                   out_ids,
                                                   &workspace_size,
                                                   &executor),
      "recurrent_gated_delta_rule: failed to get workspace size");

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "recurrent_gated_delta_rule: failed to allocate workspace");
  }

  CHECK_ACL_SUCCESS(aclnnRecurrentGatedDeltaRule(
                        workspace_addr, workspace_size, executor, stream),
                    "recurrent_gated_delta_rule: failed to perform recurrent "
                    "gated delta rule");

  aclDestroyTensor(query_ids);
  aclDestroyTensor(key_ids);
  aclDestroyTensor(value_ids);
  aclDestroyTensor(state_ids);
  aclDestroyTensor(out_ids);

  if (beta_ids != nullptr) {
    aclDestroyTensor(beta_ids);
  }
  if (actual_seq_lengths_ids != nullptr) {
    aclDestroyTensor(actual_seq_lengths_ids);
  }
  if (ssm_state_indices_ids != nullptr) {
    aclDestroyTensor(ssm_state_indices_ids);
  }
  if (num_accepted_tokens_ids != nullptr) {
    aclDestroyTensor(num_accepted_tokens_ids);
  }
  if (g_ids != nullptr) {
    aclDestroyTensor(g_ids);
  }
  if (gk_ids != nullptr) {
    aclDestroyTensor(gk_ids);
  }

  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "recurrent_gated_delta_rule: failed to free workspace");
  }

  return out_result;
}

}  // namespace xllm::kernel::npu