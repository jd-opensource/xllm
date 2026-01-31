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
                   bool enable_cuda_graph,
                   const std::optional<torch::Tensor>& mask) {
  // Optional custom mask (e.g. for Qwen2_5_VL padding) -> FlashInfer packed
  // format.
  std::optional<torch::Tensor> processed_mask;
  std::optional<torch::Tensor> mask_indptr_opt;
  if (mask.has_value()) {
    auto m = mask.value();
    if (m.defined() && m.numel() > 0) {
      auto device = query.device();
      if (m.device() != device) {
        m = m.to(device);
      }
      if (!m.is_floating_point()) {
        m = m.to(torch::kFloat32);
      }

      int64_t seq_len = m.size(0);
      // causal AND padding: attend only where j<=i and m[j]==1
      auto causal_mask = torch::tril(torch::ones(
          {seq_len, seq_len},
          torch::TensorOptions().dtype(torch::kFloat32).device(device)));
      auto combined_mask =
          causal_mask * m.unsqueeze(0).expand({seq_len, seq_len});

      const int64_t n = seq_len * seq_len;
      const int64_t num_bytes = (n + 7) / 8;
      // Pack to uint8 bitmap (8 bits/byte) for FlashInfer
      auto flat = combined_mask.contiguous().view({-1});
      if (flat.device().type() != torch::kCPU) {
        flat = flat.cpu();
      }
      auto packed = torch::zeros(
          {num_bytes},
          torch::TensorOptions().dtype(torch::kUInt8).device(flat.device()));
      auto flat_acc = flat.accessor<float, 1>();
      auto packed_acc = packed.accessor<uint8_t, 1>();
      for (int64_t i = 0; i < n; ++i) {
        if (flat_acc[i] > 0.5f) {
          packed_acc[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
      }

      if (packed.device() != device) {
        packed = packed.to(device);
      }
      processed_mask = packed.contiguous();

      // mask_indptr: [0, num_bytes] for single batch (FlashInfer batch
      // boundary)
      auto mask_indptr = torch::zeros(
          {2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      mask_indptr[0] = 0;
      mask_indptr[1] = static_cast<int32_t>(num_bytes);
      mask_indptr_opt = mask_indptr;
    }
  }

  bool use_custom_mask = processed_mask.has_value();
  std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  /*use_custom_mask=*/use_custom_mask);

  if (backend == "fa2") {
    int64_t mask_mode = use_custom_mask ? 0 : 1;  // 0=CUSTOM, 1=CAUSAL

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
        /*mask_mode_code=*/mask_mode,
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_custom_mask=*/processed_mask,
        /*maybe_mask_indptr=*/mask_indptr_opt,
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
