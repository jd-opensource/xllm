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

void batch_prefill(torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   torch::Tensor paged_kv_indptr,
                   int64_t window_size_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph) {
  std::string uri = get_batch_prefill_uri(/*backend=*/"fa2",
                                          query.scalar_type(),
                                          key.scalar_type(),
                                          output.scalar_type(),
                                          q_cu_seq_lens.scalar_type(),
                                          query.size(-1),
                                          value.size(-1),
                                          /*pos_encoding_mode=*/0,
                                          /*use_sliding_window=*/false,
                                          /*use_logits_soft_cap=*/false,
                                          /*use_fp16_qk_reduction=*/false);

  torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
  torch::Tensor qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
  torch::Tensor kv_cu_seq_lens_host = kv_cu_seq_lens.to(torch::kCPU);
  torch::Tensor kv_len_arr_host =
      kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
  const int64_t total_num_rows = qo_indptr_host.size(0);
  const int64_t batch_size = qo_indptr_host.size(0) - 1;

  auto lib =
      torch::DynamicLibrary(path_to_uri_so_lib(uri).c_str(), nullptr, true);
  std::string plan_schema_name = uri + "::plan";
  std::string run_schema_name = uri + "::ragged_run";

  auto plan_func = torch::Dispatcher::singleton()
                       .findSchemaOrThrow(plan_schema_name.c_str(), "")
                       .typed<torch::Tensor(torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            int64_t,
                                            int64_t,
                                            int64_t,
                                            int64_t,
                                            int64_t,
                                            bool,
                                            int64_t,
                                            int64_t,
                                            bool)>();
  torch::Tensor plan_info = plan_func.call(float_workspace_buffer,
                                           int_workspace_buffer,
                                           page_locked_int_workspace_buffer,
                                           qo_indptr_host,
                                           paged_kv_indptr_host,
                                           kv_len_arr_host,
                                           total_num_rows,
                                           batch_size,
                                           query.size(1),  // num_qo_heads
                                           key.size(1),    // num_kv_heads
                                           /*page_size=*/1,
                                           enable_cuda_graph,
                                           query.size(-1),  // head_dim_qk
                                           value.size(-1),  // head_dim_vo
                                           /*causal=*/true);

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
                                  std::optional<torch::Tensor>,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  bool,
                                  std::optional<torch::Tensor>,
                                  std::optional<torch::Tensor>,
                                  std::optional<torch::Tensor>,
                                  std::optional<torch::Tensor>,
                                  std::optional<torch::Tensor>,
                                  std::optional<torch::Tensor>,
                                  double,
                                  double,
                                  double,
                                  double,
                                  int64_t)>();
  run_func.call(float_workspace_buffer,
                int_workspace_buffer,
                plan_info,
                query,
                key,
                value,
                q_cu_seq_lens,
                paged_kv_indptr,
                output,
                output_lse,
                /*mask_mode_code=CAUSAL*/ 1,
                /*kv_layout_code=*/0,  // NHD layout
                window_size_left,
                support_pdl(),
                /*maybe_custom_mask=*/std::optional<torch::Tensor>(),
                /*maybe_mask_indptr=*/std::optional<torch::Tensor>(),
                /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
                /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
                /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
                /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
                /*logits_soft_cap=*/0.0,
                sm_scale,
                /*rope_rcp_scale=*/1.0,
                /*rope_rcp_theta=*/10000.0,
                /*token_pos_in_items_len=*/0);
}

}  // namespace xllm::kernel::cuda