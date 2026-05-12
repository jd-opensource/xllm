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

#include <torch/library.h>

#include "core/kernels/npu/pytorch_npu_helper.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

at::Tensor sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  at::Tensor output;
  if (cu_seqlens_q.has_value()) {
    output = torch::zeros(
        {1024},
        torch::dtype(torch::kInt32).device(cu_seqlens_q.value().device()));
  } else {
    output = torch::zeros({1024}, torch::dtype(torch::kInt32).device("npu"));
  }

  // convert str
  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());

  EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata,
               cu_seqlens_q,
               cu_seqlens_ori_kv,
               cu_seqlens_cmp_kv,
               seqused_q,
               seqused_kv,
               num_heads_q,
               num_heads_kv,
               head_dim,
               batch_size,
               max_seqlen_q,
               max_seqlen_kv,
               ori_topk,
               cmp_topk,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_win_left,
               ori_win_right,
               layout_q_ptr,
               layout_kv_ptr,
               has_ori_kv,
               has_cmp_kv,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
