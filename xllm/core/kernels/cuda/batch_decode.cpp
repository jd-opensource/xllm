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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

void batch_decode(torch::Tensor& float_workspace_buffer,
                  torch::Tensor& int_workspace_buffer,
                  torch::Tensor& page_locked_int_workspace_buffer,
                  const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  const torch::Tensor& q_cu_seq_lens,
                  const torch::Tensor& paged_kv_indptr,
                  const torch::Tensor& paged_kv_indices,
                  const torch::Tensor& paged_kv_last_page_len,
                  int64_t window_size_left,
                  torch::Tensor& output,
                  std::optional<torch::Tensor>& output_lse,
                  bool enable_cuda_graph) {
  std::string uri = get_batch_decode_uri(query.scalar_type(),
                                         k_cache.scalar_type(),
                                         output.scalar_type(),
                                         paged_kv_indptr.scalar_type(),
                                         query.size(-1),
                                         v_cache.size(-1),
                                         /*pos_encoding_mode=*/0,
                                         /*use_sliding_window=*/false,
                                         /*use_logits_soft_cap=*/false);

  torch::Tensor qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
  const int64_t batch_size = q_cu_seq_lens.size(0) - 1;
  const double sm_scale = compute_sm_scale(query.size(-1));

  // plan decode
  auto plan_info = get_module(uri)->GetFunction("plan").value()(
      to_ffi_tensor(float_workspace_buffer),
      to_ffi_tensor(int_workspace_buffer),
      to_ffi_tensor(page_locked_int_workspace_buffer),
      to_ffi_tensor(qo_indptr_host),
      batch_size,
      query.size(1),    // num_qo_heads
      k_cache.size(2),  // num_kv_heads
      k_cache.size(1),  // block_size
      enable_cuda_graph,
      window_size_left,
      /* logits_soft_cap=*/0.0,
      query.size(-1),    // head_dim_qk
      v_cache.size(-1),  // head_dim_vo
      /*empty_q_data=*/to_ffi_tensor(torch::Tensor()),
      /*empty_kv_data=*/to_ffi_tensor(torch::Tensor()));

  // batch decode
  get_module(uri)->GetFunction("run").value()(
      to_ffi_tensor(float_workspace_buffer),
      to_ffi_tensor(int_workspace_buffer),
      plan_info,
      to_ffi_tensor(query),
      to_ffi_tensor(k_cache),
      to_ffi_tensor(v_cache),
      to_ffi_tensor(paged_kv_indptr),
      to_ffi_tensor(paged_kv_indices),
      to_ffi_tensor(paged_kv_last_page_len),
      to_ffi_tensor(output),
      to_ffi_tensor(output_lse.value_or(torch::Tensor())),
      /*kv_layout_code=*/0,  // NHD layout
      window_size_left,
      support_pdl(),
      /*maybe_alibi_slopes=*/ffi::Tensor(),
      /*logits_soft_cap=*/0.0,
      /*sm_scale=*/sm_scale,
      /*rope_rcp_scale=*/1.0,
      /*rope_rcp_theta=*/1.0 / 10000.0);
}

}  // namespace xllm::kernel::cuda