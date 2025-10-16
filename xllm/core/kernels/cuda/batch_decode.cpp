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

void batch_decode(torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_size_left,
                  double sm_scale,
                  torch::Tensor output,
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

  torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
  const int64_t batch_size = paged_kv_last_page_len.size(0);

  torch::Tensor empty_q_data =
      torch::empty({0}, torch::TensorOptions().dtype(query.scalar_type()));
  torch::Tensor empty_kv_data =
      torch::empty({0}, torch::TensorOptions().dtype(k_cache.scalar_type()));

  auto lib =
      torch::DynamicLibrary(path_to_uri_so_lib(uri).c_str(), nullptr, true);
  std::string plan_schema_name = uri + "::plan";
  std::string run_schema_name = uri + "::run";

  auto plan_func = torch::Dispatcher::singleton()
                       .findSchemaOrThrow(plan_schema_name.c_str(), "")
                       .typed<torch::Tensor(torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            int64_t,
                                            int64_t,
                                            int64_t,
                                            int64_t,
                                            bool,
                                            int64_t,
                                            double,
                                            int64_t,
                                            int64_t,
                                            torch::Tensor,
                                            torch::Tensor)>();
  torch::Tensor plan_info = plan_func.call(float_workspace_buffer,
                                           int_workspace_buffer,
                                           page_locked_int_workspace_buffer,
                                           paged_kv_indptr_host,
                                           batch_size,
                                           query.size(1),    // num_qo_heads
                                           k_cache.size(2),  // num_kv_heads
                                           k_cache.size(1),  // block_size
                                           enable_cuda_graph,
                                           window_size_left,
                                           /*logits_soft_cap=*/0.0,
                                           query.size(-1),    // head_dim_qk
                                           v_cache.size(-1),  // head_dim_vo
                                           empty_q_data,
                                           empty_kv_data);

  auto run_func = torch::Dispatcher::singleton()
                      .findSchemaOrThrow(run_schema_name.c_str(), "")
                      .typed<void(torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  std::optional<torch::Tensor>,
                                  int64_t,
                                  int64_t,
                                  bool,
                                  std::optional<torch::Tensor>,
                                  double,
                                  double,
                                  double,
                                  double)>();
  run_func.call(float_workspace_buffer,
                int_workspace_buffer,
                plan_info,
                query,
                k_cache,
                v_cache,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                output,
                output_lse,
                /*kv_layout_code=*/0,  // NHD layout
                window_size_left,
                support_pdl(),
                /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
                /*logits_soft_cap=*/0.0,
                sm_scale,
                /*rope_rcp_scale=*/1.0,
                /*rope_rcp_theta=*/10000.0);
}

}  // namespace xllm::kernel::cuda