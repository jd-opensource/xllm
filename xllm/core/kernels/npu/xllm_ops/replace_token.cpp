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
#include "aclnn_replace_token.h"
#include "replace_token.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)
namespace xllm_ops {

void replace_token(torch::Tensor& dst, torch::Tensor& src) {
  xllm_ops_utils::check_tensor(dst, "dst", "replace_token");
  xllm_ops_utils::check_tensor(src, "src", "replace_token");
  aclTensor* dst_ids = nullptr;
  aclTensor* src_ids = nullptr;
  int32_t device_id = dst.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&dst_ids, dst);
  xllm_ops_utils::create_acltensor(&src_ids, src);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor;
  CHECK_ACL_SUCCESS(aclnnReplaceTokenGetWorkspaceSize(
                        dst_ids, src_ids, dst_ids, &workspace_size, &executor),
                    "replace_token: failed to get workspace size");
  void* workspace_addr = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnReplaceToken(workspace_addr, workspace_size, executor, stream),
      "replace_token: failed to replace token");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "replace_token: failed to synchronize stream");
  aclDestroyTensor(dst_ids);
  aclDestroyTensor(src_ids);
}
}  // namespace xllm_ops