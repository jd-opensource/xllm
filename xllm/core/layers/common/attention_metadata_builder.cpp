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

#include "attention_metadata_builder.h"

#include "attention_metadata.h"
#include "core/common/global_flags.h"
#include "framework/model/model_input_params.h"

namespace xllm::layer {

AttentionMetadata AttentionMetadataBuilder::build(
    const ModelInputParams& params,
    const std::optional<torch::Tensor>& attn_mask) {
  return AttentionMetadataBuilder::build(params, "float", attn_mask);
}

AttentionMetadata AttentionMetadataBuilder::build(
    const ModelInputParams& params,
    const std::string& compute_dtype,
    const std::optional<torch::Tensor>& attn_mask) {
  AttentionMetadata attn_metadata;
  attn_metadata.q_cu_seq_lens = params.q_seq_lens;
  attn_metadata.kv_cu_seq_lens = params.kv_seq_lens;
  attn_metadata.max_query_len = params.q_max_seq_len;
  attn_metadata.max_seq_len = params.kv_max_seq_len;
  attn_metadata.slot_mapping = params.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;

  // for flashinfer
  attn_metadata.paged_kv_indptr = params.paged_kv_indptr;
  attn_metadata.paged_kv_indices = params.paged_kv_indices;
  attn_metadata.paged_kv_last_page_len = params.paged_kv_last_page_len;
#if defined(USE_CUDA) || defined(USE_MUSA)
  attn_metadata.plan_info = std::make_shared<PlanInfo>();
#endif

#if defined(USE_CUDA) || defined(USE_NPU)
  // Use explicit attn_mask if provided; otherwise fall back to
  // graph_buffer.attn_mask (e.g. Qwen2_5_VL sets graph_buffer.attn_mask for
  // LongCat text encoding)
  std::optional<torch::Tensor> mask_to_use = attn_mask;
  if (!mask_to_use.has_value() && params.graph_buffer.attn_mask.defined()) {
    mask_to_use = params.graph_buffer.attn_mask;
  }
  if (mask_to_use.has_value()) {
    attn_metadata.attn_mask = mask_to_use.value();
  }
#endif

#if defined(USE_NPU)
  // Determine if we should use ACL graph mode:
  // - FLAGS_enable_graph must be enabled
  // - Must be decode phase (not prefill)
  // - tiling_data must be available
  bool is_decode = !params.batch_forward_type.is_prefill() &&
                   !params.batch_forward_type.is_mixed() &&
                   !params.batch_forward_type.is_chunked_prefill();
  bool use_acl_graph = FLAGS_enable_graph && is_decode &&
                       params.graph_buffer.tiling_data.defined();
  if (use_acl_graph) {
    // ACL graph mode: use CustomPagedAttention with tiling_data on device
    attn_metadata.paged_attention_tiling_data = params.graph_buffer.tiling_data;
  }
  // Provide host seq_lens for NPU kernels (required by CustomPagedAttention).
  if (!params.kv_seq_lens_vec.empty()) {
    attn_metadata.kv_seq_lens_host =
        torch::tensor(params.kv_seq_lens_vec, torch::kInt);
  }
#endif
  attn_metadata.is_chunked_prefill =
      params.batch_forward_type.is_mixed() ||
      params.batch_forward_type.is_chunked_prefill();
  attn_metadata.is_prefill = params.batch_forward_type.is_prefill();
  if (!attn_metadata.is_prefill || FLAGS_enable_mla) {
    attn_metadata.block_table = params.block_tables;
#if defined(USE_NPU)
    // NPU path uses per-sequence lengths (not cumulative), so no diff.
    attn_metadata.kv_seq_lens = params.kv_seq_lens;
#else
    attn_metadata.kv_seq_lens = torch::diff(params.kv_seq_lens);  // kv seqlens
#endif
  }

  attn_metadata.is_dummy = (params.q_max_seq_len == 0);
  if (attn_metadata.is_dummy) {
    attn_metadata.slot_mapping =
        torch::tensor({1}, params.new_cache_slots.options());
  }

  // Set is_causal: true for prefill (causal attention), false for decode
  // (non-causal) Default to true (causal) if not explicitly set
  attn_metadata.is_causal =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  // Copy enable_cuda_graph flag from params
  attn_metadata.enable_cuda_graph = params.enable_cuda_graph;

#if defined(USE_ILU)
  attn_metadata.block_table = params.block_tables;
#endif

  // TODO: set use_tensor_core from options.
  // for xattention
  if (params.has_llmrec_params()) {
    const auto& llmrec_params = *params.llmrec_params();
    if (llmrec_params.current_round_tensor.defined() &&
        llmrec_params.current_round_tensor.numel() > 0) {
      attn_metadata.step_tensor = llmrec_params.current_round_tensor;
    }
  }

  return attn_metadata;
}

}  // namespace xllm::layer
