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

#include "function_factory.h"
#include "musa_ops_api.h"

namespace xllm::kernel::musa {

void batch_prefill(torch::Tensor& float_workspace_buffer,
                   torch::Tensor& int_workspace_buffer,
                   torch::Tensor& page_locked_int_workspace_buffer,
                   torch::Tensor& query,
                   torch::Tensor& key,
                   torch::Tensor value,
                   const torch::Tensor& q_cu_seq_lens,
                   const torch::Tensor& kv_cu_seq_lens,
                   int64_t max_seqlen_q,
                   int64_t max_seqlen_kv,
                   double sm_scale,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph) {
  torch::Tensor lse_mate, temp_a, temp_b;
  std::tie(output, lse_mate, temp_a, temp_b) =
      FunctionFactory::get_instance().mate_func().call(
          query,
          key,
          value,
          /*k_new=*/std::nullopt,
          /*v_new=*/std::nullopt,
          /*q_v=*/std::nullopt,
          output,
          q_cu_seq_lens,
          kv_cu_seq_lens,
          /*cu_seqlens_k_new=*/std::nullopt,
          /*seqused_q=*/std::nullopt,
          /*seqused_k=*/std::nullopt,
          max_seqlen_q,
          max_seqlen_kv,
          /*page_table=*/std::nullopt,
          /*kv_batch_idx=*/std::nullopt,
          /*leftpad_k=*/std::nullopt,
          /*rotary_cos=*/std::nullopt,
          /*rotary_sin=*/std::nullopt,
          /*seqlens_rotary=*/std::nullopt,
          /*q_descale=*/std::nullopt,
          /*k_descale=*/std::nullopt,
          /*v_descale=*/std::nullopt,
          /*softmax_scale=*/std::nullopt,
          /*is_causal=*/true,
          /*window_size_left=*/-1,
          /*window_size_right=*/-1,
          /*attention_chunk=*/0,
          /*softcap=*/0.f,
          /*is_rotary_interleaved=*/false,
          /*scheduler_metadata=*/std::nullopt,
          /*num_splits=*/1,
          /*pack_gqa=*/std::nullopt,
          /*sm_margin=*/0);
}

}  // namespace xllm::kernel::musa
