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
#include "function_factory.h"

namespace xllm::kernel::cuda {

void batch_prefill(const std::string& uri,
                   torch::Tensor plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph) {
  std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  /*use_custom_mask=*/false);

  if (backend == "fa2") {
    FunctionFactory::get_instance().fa2_prefill_ragged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        key,
        value,
        q_cu_seq_lens,
        kv_cu_seq_lens,
        output,
        output_lse,
        /*mask_mode_code=*/1,  // CAUSAL
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
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
        /*rope_rcp_theta=*/1.0 / 10000.0,
        /*token_pos_in_items_len=*/0);
  } else if (backend == "fa3") {
    FunctionFactory::get_instance().fa3_prefill_ragged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        key,
        value,
        q_cu_seq_lens,
        kv_cu_seq_lens,
        output,
        output_lse,
        /*mask_mode_code=*/1,  // CAUSAL
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*token_pos_in_items_len=*/0);
  }
}

}  // namespace xllm::kernel::cuda
