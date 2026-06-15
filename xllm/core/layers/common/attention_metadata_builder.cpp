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

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "attention_metadata.h"
#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_types.h"
#include "runtime/forward_params.h"

namespace xllm::layer {

namespace {

torch::TensorOptions int32_options_like(const torch::Tensor& preferred,
                                        const torch::Tensor& fallback) {
  if (preferred.defined()) {
    return preferred.options().dtype(torch::kInt32);
  }
  if (fallback.defined()) {
    return fallback.options().dtype(torch::kInt32);
  }
  return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
}

AttentionMetadata build_attention_metadata(
    const BatchInputMeta& meta,
    const AttentionInput& attention,
    const GraphInput& graph,
    const LlmRecMultiRoundParams* llmrec_params,
    bool enable_cuda_graph,
    bool enable_mla,
    const std::string& compute_dtype,
    const std::optional<torch::Device>& device,
    const std::optional<torch::Tensor>& attn_mask) {
  // MLA mode still affects which shared tensors must be materialized for
  // attention execution, but the flag itself is no longer carried in metadata.
  AttentionMetadata attn_metadata;
  attn_metadata.q_cu_seq_lens = attention.device.q_seq_lens;
  attn_metadata.kv_cu_seq_lens = attention.device.kv_seq_lens;
  attn_metadata.max_query_len = meta.q_max_seq_len;
  attn_metadata.max_seq_len = meta.kv_max_seq_len;
  if (!attention.host.kv_seq_lens.empty()) {
    const bool is_cu_seq_lens =
        attention.host.kv_seq_lens.size() ==
            static_cast<size_t>(meta.num_sequences + 1) &&
        attention.host.kv_seq_lens.front() == 0;
    attn_metadata.total_kv_len =
        is_cu_seq_lens ? attention.host.kv_seq_lens.back()
                       : std::accumulate(attention.host.kv_seq_lens.begin(),
                                         attention.host.kv_seq_lens.end(),
                                         int64_t{0});
  }
  attn_metadata.kv_seq_lens_vec = attention.host.kv_seq_lens;
  attn_metadata.q_seq_lens_vec = attention.host.q_seq_lens;
  attn_metadata.slot_mapping = attention.device.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;

  // for flashinfer
  attn_metadata.paged_kv_indptr = attention.device.paged_kv_indptr;
  attn_metadata.paged_kv_indices = attention.device.paged_kv_indices;
  attn_metadata.paged_kv_last_page_len =
      attention.device.paged_kv_last_page_len;
#if defined(USE_CUDA) || defined(USE_MUSA)
  attn_metadata.plan_info = std::make_shared<PlanInfo>();
  attn_metadata.shared_plan_info = std::make_shared<PlanInfo>();
  attn_metadata.unshared_plan_info = std::make_shared<PlanInfo>();
#endif

#if defined(USE_CUDA) || defined(USE_NPU) || defined(USE_MLU)
  // Use explicit attn_mask if provided; otherwise fall back to
  // graph_buffer.attn_mask (e.g. Qwen2_5_VL sets graph_buffer.attn_mask for
  // LongCat text encoding)
  std::optional<torch::Tensor> mask_to_use = attn_mask;
  if (!mask_to_use.has_value() && graph.attn_mask.defined()) {
    mask_to_use = graph.attn_mask;
  }
  if (mask_to_use.has_value()) {
    attn_metadata.attn_mask = mask_to_use.value();
  }
#endif

#if defined(USE_NPU)
  // Determine if we should use ACL graph mode:
  // - --enable_graph=true
  // - Must be decode phase (not prefill)
  // - tiling_data must be available
  bool is_decode = !meta.batch_forward_type.is_prefill() &&
                   !meta.batch_forward_type.is_mixed() &&
                   !meta.batch_forward_type.is_chunked_prefill();
  bool use_acl_graph = ::xllm::ExecutionConfig::get_instance().enable_graph() &&
                       is_decode && graph.tiling_data.defined();
  if (use_acl_graph) {
    // ACL graph mode: use CustomPagedAttention with tiling_data on device
    attn_metadata.paged_attention_tiling_data = graph.tiling_data;
  }
  // Provide host seq_lens for NPU kernels (required by CustomPagedAttention).
  if (!attention.host.kv_seq_lens.empty()) {
    attn_metadata.kv_seq_lens_host =
        torch::tensor(attention.host.kv_seq_lens, torch::kInt);
  }
#endif
  attn_metadata.is_chunked_prefill =
      meta.batch_forward_type.is_mixed() ||
      meta.batch_forward_type.is_chunked_prefill();
  attn_metadata.is_prefill = meta.batch_forward_type.is_prefill();

  // MLA-family MLU paths require per-sequence q/kv lengths during prefill.
  if (!attn_metadata.is_prefill || enable_mla) {
    attn_metadata.block_table = attention.device.block_tables;
#if !defined(USE_NPU) && !defined(USE_CUDA)
    attn_metadata.kv_seq_lens =
        torch::diff(attention.device.kv_seq_lens);  // kv seqlens
    attn_metadata.q_seq_lens =
        torch::diff(attention.device.q_seq_lens);  // q seqlens
#endif
  }
#if defined(USE_NPU)
  // NPU path uses per-sequence lengths (not cumulative), so no diff.
  // Ensure per-sequence lengths are available for NPU kernels in all phases.
  if (attention.device.kv_seq_lens.defined()) {
    attn_metadata.kv_seq_lens = attention.device.kv_seq_lens;
  }
  if (attention.device.q_seq_lens.defined()) {
    attn_metadata.q_seq_lens = attention.device.q_seq_lens;
    CHECK(attention.device.q_cu_seq_lens.defined())
        << "q_cu_seq_lens must be provided by upstream";
    auto zero = torch::zeros({1}, attention.device.q_cu_seq_lens.options());
    attn_metadata.q_cu_seq_lens =
        torch::cat({zero, attention.device.q_cu_seq_lens}, 0);
  }
#endif

  attn_metadata.is_dummy = (meta.q_max_seq_len == 0);
  if (attn_metadata.is_dummy) {
    torch::TensorOptions options = int32_options_like(
        attention.device.new_cache_slots, attention.device.q_seq_lens);
    if (!attention.device.new_cache_slots.defined() &&
        !attention.device.q_seq_lens.defined()) {
      CHECK(device.has_value())
          << "dummy attention requires device when new_cache_slots is "
             "undefined";
      options = options.device(device.value());
    }
    attn_metadata.slot_mapping = torch::tensor({1}, options);
    attn_metadata.q_cu_seq_lens = torch::tensor({0, 1}, options);
    attn_metadata.q_seq_lens = torch::tensor({1}, options);
    attn_metadata.kv_seq_lens = torch::tensor({1}, options);
    attn_metadata.max_query_len = 1;
    attn_metadata.max_seq_len = std::max<int64_t>(attn_metadata.max_seq_len, 1);
  }

  // Set is_causal: true for prefill (causal attention), false for decode
  // (non-causal) Default to true (causal) if not explicitly set
  attn_metadata.is_causal =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  attn_metadata.enable_cuda_graph = enable_cuda_graph;

#if defined(USE_CUDA) || defined(USE_MUSA)
  if (attn_metadata.is_causal && !attn_metadata.enable_cuda_graph) {
    attn_metadata.qo_indptr = attn_metadata.q_cu_seq_lens.to(torch::kCUDA);
  }
#endif

#if defined(USE_ILU)
  attn_metadata.block_table = attention.device.block_tables;
#endif

  // TODO: set use_tensor_core from options.
  // for xattention
  if (llmrec_params != nullptr) {
    if (llmrec_params->current_round_tensor.defined() &&
        llmrec_params->current_round_tensor.numel() > 0) {
      attn_metadata.step_tensor = llmrec_params->current_round_tensor;
    }

    if (!::xllm::RecConfig::get_instance().enable_xattention_one_stage()) {
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.xattention_two_stage_decode_cache.emplace(
          XAttentionTwoStageDecodeCache{});
      auto& cache = attn_metadata.xattention_two_stage_decode_cache.value();

      cache.shared_lse = llmrec_params->two_stage_shared_lse;
      cache.shared_o = llmrec_params->two_stage_shared_o;
      cache.unshared_lse = llmrec_params->two_stage_unshared_lse;
      cache.unshared_o = llmrec_params->two_stage_unshared_o;
      cache.q_cu_seq_lens_shared =
          llmrec_params->two_stage_q_cu_seq_lens_shared;
      cache.qo_indptr_expanded = llmrec_params->two_stage_qo_indptr_expanded;
      cache.paged_kv_indptr_expanded =
          llmrec_params->two_stage_paged_kv_indptr_expanded;
      cache.paged_kv_indices_expanded =
          llmrec_params->two_stage_paged_kv_indices_expanded;
      cache.paged_kv_last_page_len_expanded =
          llmrec_params->two_stage_paged_kv_last_page_len_expanded;

      if (cache.q_cu_seq_lens_shared.defined()) {
        cache.cached_batch_size =
            static_cast<int32_t>(cache.q_cu_seq_lens_shared.numel()) - 1;
      }
      cache.cached_beam_size = llmrec_params->beam_width;
      if (!llmrec_params->unshared_k_caches.empty()) {
        cache.cached_max_decode_step =
            static_cast<int32_t>(llmrec_params->unshared_k_caches[0].size(2));
      }
      if (cache.shared_o.defined() && cache.shared_o.dim() == 3) {
        cache.cached_num_heads = static_cast<int32_t>(cache.shared_o.size(1));
        cache.cached_head_size = static_cast<int32_t>(cache.shared_o.size(2));
      }
      if (llmrec_params->current_round_tensor.defined() &&
          llmrec_params->current_round_tensor.numel() > 0) {
        cache.cached_step = llmrec_params->current_round_tensor.item<int32_t>();
      }
#endif
    }
  }

  return attn_metadata;
}

}  // namespace

AttentionMetadata AttentionMetadataBuilder::build(
    const ForwardInput& input,
    bool enable_mla,
    const std::optional<torch::Tensor>& attn_mask,
    const std::optional<torch::Device>& device) {
  return AttentionMetadataBuilder::build(
      input, enable_mla, "float", attn_mask, device);
}

AttentionMetadata AttentionMetadataBuilder::build(
    const ForwardInput& input,
    bool enable_mla,
    const std::string& compute_dtype,
    const std::optional<torch::Tensor>& attn_mask,
    const std::optional<torch::Device>& device) {
  return AttentionMetadataBuilder::build(
      input.meta,
      input.attention,
      input.graph,
      input.llmrec_params(),
      input.enable_cuda_graph || input.enable_graph,
      enable_mla,
      compute_dtype,
      attn_mask,
      device);
}

AttentionMetadata AttentionMetadataBuilder::build(
    const BatchInputMeta& meta,
    const AttentionInput& attention,
    const GraphInput& graph,
    const LlmRecMultiRoundParams* llmrec_params,
    bool enable_cuda_graph,
    bool enable_mla,
    const std::optional<torch::Tensor>& attn_mask,
    const std::optional<torch::Device>& device) {
  return AttentionMetadataBuilder::build(meta,
                                         attention,
                                         graph,
                                         llmrec_params,
                                         enable_cuda_graph,
                                         enable_mla,
                                         "float",
                                         attn_mask,
                                         device);
}

AttentionMetadata AttentionMetadataBuilder::build(
    const BatchInputMeta& meta,
    const AttentionInput& attention,
    const GraphInput& graph,
    const LlmRecMultiRoundParams* llmrec_params,
    bool enable_cuda_graph,
    bool enable_mla,
    const std::string& compute_dtype,
    const std::optional<torch::Tensor>& attn_mask,
    const std::optional<torch::Device>& device) {
  return build_attention_metadata(meta,
                                  attention,
                                  graph,
                                  llmrec_params,
                                  enable_cuda_graph,
                                  enable_mla,
                                  compute_dtype,
                                  device,
                                  attn_mask);
}

}  // namespace xllm::layer
