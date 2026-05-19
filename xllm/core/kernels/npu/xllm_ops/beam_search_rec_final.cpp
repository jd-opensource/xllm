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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "acl/acl.h"
#include "aclnn_onerec_final_beam_select.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

void run_beam_search_rec_final(const torch::Tensor& logprobs,
                               const torch::Tensor& top_tokens,
                               const torch::Tensor& top_logprobs,
                               torch::Tensor& sequence_group,
                               int64_t current_step,
                               int64_t result_width,
                               torch::Tensor& out_token_ids,
                               torch::Tensor& out_token_index,
                               torch::Tensor& out_log_probs,
                               torch::Tensor& out_beam_count_prefix_sums,
                               torch::Tensor& out_sequence) {
  CHECK_GT(result_width, 0)
      << "beam_search_rec final select requires positive result_width";
  CHECK_EQ(out_sequence.dim(), 3)
      << "beam_search_rec final select expects 3D out_sequence";
  CHECK_EQ(out_sequence.size(1), result_width)
      << "beam_search_rec final select output width mismatch";
  check_tensor(logprobs, "logprobs", "beam_search_rec_final");
  check_tensor(top_tokens, "top_tokens", "beam_search_rec_final");
  check_tensor(top_logprobs, "top_logprobs", "beam_search_rec_final");
  check_tensor(sequence_group, "sequence_group", "beam_search_rec_final");
  check_tensor(out_token_ids, "out_token_ids", "beam_search_rec_final");
  check_tensor(out_token_index, "out_token_index", "beam_search_rec_final");
  check_tensor(out_log_probs, "out_log_probs", "beam_search_rec_final");
  check_tensor(out_beam_count_prefix_sums,
               "out_beam_count_prefix_sums",
               "beam_search_rec_final");
  check_tensor(out_sequence, "out_sequence", "beam_search_rec_final");

  // The final widened result is returned to host only, so cache-select prefix
  // metadata is not consumed after this kernel.
  out_beam_count_prefix_sums.zero_();

  aclTensor* logprobs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_logprobs_ids = nullptr;
  aclTensor* sequence_group_ids = nullptr;
  aclTensor* out_token_ids_ids = nullptr;
  aclTensor* out_token_index_ids = nullptr;
  aclTensor* out_log_probs_ids = nullptr;
  aclTensor* out_sequence_ids = nullptr;
  int32_t device_id = logprobs.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  create_acltensor(&logprobs_ids, logprobs);
  create_acltensor(&top_tokens_ids, top_tokens);
  create_acltensor(&top_logprobs_ids, top_logprobs);
  create_acltensor(&sequence_group_ids, sequence_group);
  create_acltensor(&out_token_ids_ids, out_token_ids);
  create_acltensor(&out_token_index_ids, out_token_index);
  create_acltensor(&out_log_probs_ids, out_log_probs);
  create_acltensor(&out_sequence_ids, out_sequence);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnOnerecFinalBeamSelectGetWorkspaceSize(logprobs_ids,
                                                 top_tokens_ids,
                                                 top_logprobs_ids,
                                                 sequence_group_ids,
                                                 current_step,
                                                 out_token_ids_ids,
                                                 out_token_index_ids,
                                                 out_log_probs_ids,
                                                 out_sequence_ids,
                                                 &workspace_size,
                                                 &executor),
      "beam_search_rec_final: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search_rec_final: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(aclnnOnerecFinalBeamSelect(
                        workspace_addr, workspace_size, executor, stream),
                    "beam_search_rec_final: failed to execute kernel");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "beam_search_rec_final: failed to synchronize stream");

  aclDestroyTensor(logprobs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_logprobs_ids);
  aclDestroyTensor(sequence_group_ids);
  aclDestroyTensor(out_token_ids_ids);
  aclDestroyTensor(out_token_index_ids);
  aclDestroyTensor(out_log_probs_ids);
  aclDestroyTensor(out_sequence_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "beam_search_rec_final: failed to free workspace");
  }
}

}  // namespace xllm::kernel::npu
