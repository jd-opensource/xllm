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

#include "xattention.h"

#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/xattention/xattention_ops_api.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

XAttentionImpl::XAttentionImpl(int64_t num_heads,
                               int64_t head_size,
                               float scale,
                               int64_t num_kv_heads,
                               int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> XAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& output,
    KVCache& kv_cache) {
  std::optional<at::Tensor> output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";

  if (attn_metadata.is_prefill) {
    prefill_forward(attn_metadata, query, key, value, output, output_lse);
  } else {
    decoder_forward(attn_metadata, query, key, value, output);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void XAttentionImpl::prefill_forward(const AttentionMetadata& attn_metadata,
                                     torch::Tensor& query,
                                     torch::Tensor& key,
                                     torch::Tensor& value,
                                     torch::Tensor& output,
                                     std::optional<at::Tensor>& output_lse) {
  flashinfer::update_prefill_plan_info(
      attn_metadata.plan_info,
      xllm::kernel::cuda::determine_attention_backend(
          /*pos_encoding_mode=*/0,
          /*use_fp16_qk_reduction=*/false,
          /*use_custom_mask=*/false,
          /*causal=*/true),
      attn_metadata,
      query.scalar_type(),
      key.scalar_type(),
      output.scalar_type(),
      head_size_,
      head_size_,
      num_heads_,
      num_kv_heads_,
      /*enable_cuda_graph=*/false);

  xllm::kernel::cuda::prefill_reshape_and_cache(
      key, value, attn_metadata.full_k_cache, attn_metadata.full_v_cache);

  torch::Tensor float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  torch::Tensor int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  torch::Tensor page_locked_int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  xllm::kernel::cuda::batch_prefill_with_optional_piecewise_capture(
      attn_metadata.plan_info->uri,
      attn_metadata.plan_info->plan_info,
      float_workspace_buffer,
      int_workspace_buffer,
      page_locked_int_workspace_buffer,
      query,
      key,
      value,
      attn_metadata.q_cu_seq_lens,
      attn_metadata.kv_cu_seq_lens,
      sliding_window_,
      scale_,
      output,
      output_lse);
}

void XAttentionImpl::decoder_forward(const AttentionMetadata& attn_metadata,
                                     torch::Tensor& query,
                                     torch::Tensor& key,
                                     torch::Tensor& value,
                                     torch::Tensor& output) {
  key = key.contiguous();
  value = value.contiguous();

  xllm::kernel::cuda::decoder_reshape_and_cache(key,
                                                value,
                                                attn_metadata.unshared_k_cache,
                                                attn_metadata.unshared_v_cache,
                                                attn_metadata.block_table,
                                                attn_metadata.step_tensor);

  torch::Tensor full_k_cache = attn_metadata.full_k_cache.unsqueeze(1);
  torch::Tensor full_v_cache = attn_metadata.full_v_cache.unsqueeze(1);

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    std::string backend = "fa3";
    flashinfer::update_decode_plan_info(
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
        /*block_size=*/full_k_cache.size(1),
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/false,
        /*use_tensor_core=*/decode_use_tensor_core_);
  }

  std::optional<at::Tensor> unshared_lse = std::nullopt;

  torch::Tensor float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  torch::Tensor int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  torch::Tensor page_locked_int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  xllm::kernel::cuda::batch_decode(attn_metadata.plan_info->uri,
                                   attn_metadata.plan_info->plan_info,
                                   float_workspace_buffer,
                                   int_workspace_buffer,
                                   page_locked_int_workspace_buffer,
                                   query,
                                   full_k_cache,
                                   full_v_cache,
                                   attn_metadata.paged_kv_indptr,
                                   attn_metadata.paged_kv_indices,
                                   attn_metadata.paged_kv_last_page_len,
                                   sliding_window_,
                                   scale_,
                                   output,
                                   unshared_lse,
                                   decode_use_tensor_core_,
                                   attn_metadata.qo_indptr);
}

}  // namespace layer
}  // namespace xllm
