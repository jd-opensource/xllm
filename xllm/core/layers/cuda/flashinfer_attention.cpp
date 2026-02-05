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

#include "flashinfer_attention.h"

#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "kernels/cuda/utils.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

FlashInferAttentionImpl::FlashInferAttentionImpl(int64_t num_heads,
                                                 int64_t head_size,
                                                 float scale,
                                                 int64_t num_kv_heads,
                                                 int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window - 1) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
FlashInferAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                                 torch::Tensor& query,
                                 torch::Tensor& key,
                                 torch::Tensor& value,
                                 torch::Tensor& output,
                                 KVCache& kv_cache) {
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // maybe we need to update shared attn state before execute attention,
  // currently we update flashinfer step_wise_attn_state_ at layer 0.
  bool causal = attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  std::string backend = xllm::kernel::cuda::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      /*use_custom_mask=*/false,
      /*causal=*/causal);

  // NOTE: we only support "fa2" backend for BatchPrefillWithPagedKvcacheKernel
  // for flashinfer v0.6.2, because it would cause performance degradation if
  // using "fa3" backend.
  if (!causal) {
    backend = "fa2";
  }

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        backend,
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size=*/k_cache.size(1),
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/attn_metadata.enable_cuda_graph,
        /*causal=*/causal,
        /*use_tensor_core=*/decode_use_tensor_core_);
  }

  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key;
  reshape_paged_cache_params.value = value;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  xllm::kernel::AttentionParams attention_params(attn_metadata);
  attention_params.query = query;
  attention_params.output = output;
  attention_params.output_lse = output_lse;
  // attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  // for flashinfer
  attention_params.float_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  attention_params.int_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  attention_params.page_locked_int_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();
  attention_params.use_tensor_core = decode_use_tensor_core_;

  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";
  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    attention_params.query = query;
    attention_params.output = output;
    attention_params.k_cache = k_cache;
    attention_params.v_cache = v_cache;

    xllm::kernel::batch_decode(attention_params);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
