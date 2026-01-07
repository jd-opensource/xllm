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
#include "aclnnop/aclnn_apply_top_k_top_p.h"
#include "core/common/macros.h"
#include "top_k_top_p.h"

namespace xllm_ops {
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP) {
  xllm_ops_utils::check_tensor(logits, "logits", "top_k_top_p");
  xllm_ops_utils::check_tensor(topK, "topK", "top_k_top_p");
  xllm_ops_utils::check_tensor(topP, "topP", "top_k_top_p");
  aclTensor* logits_ids = nullptr;
  aclTensor* topK_ids = nullptr;
  aclTensor* topP_ids = nullptr;
  int32_t device_id = logits.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&logits_ids, logits);
  xllm_ops_utils::create_acltensor(&topK_ids, topK);
  xllm_ops_utils::create_acltensor(&topP_ids, topP);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnApplyTopKTopPGetWorkspaceSize(logits_ids,
                                                       topP_ids,
                                                       topK_ids,
                                                       logits_ids,
                                                       &workspace_size,
                                                       &executor),
                    "top_k_top_p: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "top_k_top_p: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnApplyTopKTopP(workspace_addr, workspace_size, executor, stream),
      "top_k_top_p: failed to apply top k top p");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "top_k_top_p: failed to synchronize stream");
  aclDestroyTensor(logits_ids);
  aclDestroyTensor(topK_ids);
  aclDestroyTensor(topP_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "top_k_top_p: failed to free workspace");
  }
}
}  // namespace xllm_ops