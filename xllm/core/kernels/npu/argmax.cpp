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
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_argmax.h"
#include "core/common/macros.h"
#include "npu_ops_api.h"
#include "utils.h"

namespace xllm::kernel::npu {
torch::Tensor argmax_int32(const torch::Tensor& input, int64_t dim) {
  check_tensor(input, "input", "argmax_int32");
  CHECK(input.device().is_privateuseone())
      << "argmax_int32 expects an NPU tensor";
  CHECK(input.is_contiguous()) << "argmax_int32 expects contiguous input";
  if (dim < 0) {
    dim += input.dim();
  }
  CHECK_GE(dim, 0);
  CHECK_LT(dim, input.dim());

  std::vector<int64_t> out_shape = input.sizes().vec();
  out_shape.erase(out_shape.begin() + dim);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  torch::Tensor output =
      torch::empty(out_shape, input.options().dtype(torch::kInt32));

  aclTensor* input_acl = nullptr;
  aclTensor* output_acl = nullptr;
  create_acltensor(&input_acl, input);
  create_acltensor(&output_acl, output);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnArgMaxGetWorkspaceSize(input_acl,
                                                dim,
                                                /*keepdim=*/false,
                                                output_acl,
                                                &workspace_size,
                                                &executor),
                    "argmax_int32: failed to get workspace size");
  at::DataPtr workspace_holder;
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace_holder =
        c10_npu::NPUCachingAllocator::get()->allocate(workspace_size);
    workspace = workspace_holder.get();
  }
  const int32_t device_id = input.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  CHECK_ACL_SUCCESS(aclnnArgMax(workspace, workspace_size, executor, stream),
                    "argmax_int32: failed to run aclnnArgMax");

  aclDestroyTensor(input_acl);
  aclDestroyTensor(output_acl);
  return output;
}

}  // namespace xllm::kernel::npu
