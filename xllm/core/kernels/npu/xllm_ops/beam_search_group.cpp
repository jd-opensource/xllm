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
#include "beam_search_group.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)
namespace xllm_ops {

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& sequence_group,
                 int64_t current_step,
                 torch::Tensor& out_token_ids,
                 torch::Tensor& out_token_index,
                 torch::Tensor& out_log_probs,
                 torch::Tensor& out_beam_count_prefix_sums,
                 torch::Tensor& out_sequence) {
  xllm_ops_utils::check_tensor(logprobs, "logprobs", "beam_search");
  xllm_ops_utils::check_tensor(top_tokens, "top_tokens", "beam_search");
  xllm_ops_utils::check_tensor(top_logprobs, "top_logprobs", "beam_search");
  xllm_ops_utils::check_tensor(sequence_group, "sequence_group", "beam_search");
  aclTensor* logprobs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_logprobs_ids = nullptr;
  aclTensor* sequence_group_ids = nullptr;
  aclTensor* out_token_ids_ids = nullptr;
  aclTensor* out_token_index_ids = nullptr;
  aclTensor* out_log_probs_ids = nullptr;
  aclTensor* out_beam_count_prefix_sums_ids = nullptr;
  aclTensor* out_sequence_ids = nullptr;
  int32_t device_id = logprobs.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&logprobs_ids, logprobs);
  xllm_ops_utils::create_acltensor(&top_tokens_ids, top_tokens);
  xllm_ops_utils::create_acltensor(&top_logprobs_ids, top_logprobs);
  xllm_ops_utils::create_acltensor(&sequence_group_ids, sequence_group);
  xllm_ops_utils::create_acltensor(&out_token_ids_ids, out_token_ids);
  xllm_ops_utils::create_acltensor(&out_token_index_ids, out_token_index);
  xllm_ops_utils::create_acltensor(&out_log_probs_ids, out_log_probs);
  xllm_ops_utils::create_acltensor(&out_beam_count_prefix_sums_ids,
                                   out_beam_count_prefix_sums);
  xllm_ops_utils::create_acltensor(&out_sequence_ids, out_sequence);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnBeamSearchGroupGetWorkspaceSize(logprobs_ids,
                                           top_tokens_ids,
                                           top_logprobs_ids,
                                           sequence_group_ids,
                                           current_step,
                                           out_token_ids_ids,
                                           out_token_index_ids,
                                           out_log_probs_ids,
                                           out_beam_count_prefix_sums_ids,
                                           out_sequence_ids,
                                           &workspace_size,
                                           &executor),
      "beam_search group: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search group: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnBeamSearchGroup(workspace_addr, workspace_size, executor, stream),
      "beam_search group: failed to perform beam search");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "beam_search group: failed to synchronize stream");
  aclDestroyTensor(logprobs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_logprobs_ids);
  aclDestroyTensor(sequence_group_ids);
  aclDestroyTensor(out_token_ids_ids);
  aclDestroyTensor(out_token_index_ids);
  aclDestroyTensor(out_log_probs_ids);
  aclDestroyTensor(out_beam_count_prefix_sums_ids);
  aclDestroyTensor(out_sequence_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "beam_search group: failed to free workspace");
  }
}
}  // namespace xllm_ops