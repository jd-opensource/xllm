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

#include <algorithm>

#include "core/common/global_flags.h"
#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/xattention/xattention_ops_api.h"
#include "kernels/ops_api.h"
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

void XAttentionImpl::run_single_stage_decode(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& key,
    torch::Tensor& query,
    torch::Tensor& output) {
  torch::Tensor full_k_cache = attn_metadata.full_k_cache.unsqueeze(1);
  torch::Tensor full_v_cache = attn_metadata.full_v_cache.unsqueeze(1);

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph "
           "is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    std::string backend = "fa3";
    flashinfer::update_plan_info(attn_metadata.plan_info,
                                 backend,
                                 attn_metadata,
                                 query.scalar_type(),
                                 key.scalar_type(),
                                 output.scalar_type(),
                                 head_size_,
                                 head_size_,
                                 num_heads_,
                                 num_kv_heads_,
                                 /*block_size*/ full_k_cache.size(1),
                                 /*window_size_left*/ sliding_window_,
                                 /*enable_cuda_graph*/ false,
                                 /*causal*/ attn_metadata.is_causal,
                                 /*use_tensor_core*/ decode_use_tensor_core_);
  }

  xllm::kernel::AttentionParams attention_params(attn_metadata);
  auto unshared_lse = std::nullopt;

  attention_params.return_lse = false;
  attention_params.output_lse = unshared_lse;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  attention_params.float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  attention_params.int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  attention_params.page_locked_int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";

  attention_params.query = query;
  attention_params.output = output;
  attention_params.k_cache = full_k_cache;
  attention_params.v_cache = full_v_cache;
  attention_params.use_tensor_core = decode_use_tensor_core_;
  xllm::kernel::batch_decode(attention_params);
}

void XAttentionImpl::run_two_stage_decode(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& output) {
  CHECK(attn_metadata.shared_plan_info != nullptr)
      << "shared_plan_info must be initialized for two-stage decode.";
  CHECK(attn_metadata.unshared_plan_info != nullptr)
      << "unshared_plan_info must be initialized for two-stage decode.";

  const int64_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
  CHECK_GT(batch_size, 0) << "invalid batch_size";

  const int64_t total_beam = query.size(0);
  CHECK_EQ(total_beam % batch_size, 0)
      << "total_beam must be divisible by batch_size";

  CHECK(attn_metadata.xattention_two_stage_decode_cache.has_value())
      << "xattention_two_stage_decode_cache must be prepared before "
         "run_two_stage_decode (expected from "
         "RecWorkerImpl::prepare_input_for_current_round).";

  const auto& cache = attn_metadata.xattention_two_stage_decode_cache.value();
  CHECK(cache.shared_lse.defined() && cache.shared_o.defined() &&
        cache.unshared_lse.defined() && cache.unshared_o.defined())
      << "two-stage output cache tensors are not initialized.";
  CHECK(cache.q_cu_seq_lens_shared.defined() &&
        cache.paged_kv_indptr_expanded.defined() &&
        cache.paged_kv_indices_expanded.defined() &&
        cache.paged_kv_last_page_len_expanded.defined())
      << "two-stage index cache tensors are not initialized.";

  CHECK_EQ(cache.shared_o.size(0), total_beam)
      << "shared_o first dim mismatch: expected total_beam=" << total_beam
      << ", got " << cache.shared_o.size(0);
  CHECK_EQ(cache.unshared_o.size(0), total_beam)
      << "unshared_o first dim mismatch: expected total_beam=" << total_beam
      << ", got " << cache.unshared_o.size(0);
  CHECK_EQ(cache.q_cu_seq_lens_shared.numel(), batch_size + 1)
      << "q_cu_seq_lens_shared size mismatch: expected " << (batch_size + 1)
      << ", got " << cache.q_cu_seq_lens_shared.numel();
  CHECK_EQ(cache.paged_kv_indptr_expanded.numel(), total_beam + 1)
      << "paged_kv_indptr_expanded size mismatch: expected " << (total_beam + 1)
      << ", got " << cache.paged_kv_indptr_expanded.numel();
  CHECK_EQ(cache.paged_kv_indices_expanded.numel(), total_beam)
      << "paged_kv_indices_expanded size mismatch: expected " << total_beam
      << ", got " << cache.paged_kv_indices_expanded.numel();
  CHECK_EQ(cache.paged_kv_last_page_len_expanded.numel(), total_beam)
      << "paged_kv_last_page_len_expanded size mismatch: expected "
      << total_beam << ", got "
      << cache.paged_kv_last_page_len_expanded.numel();

  // ===== Shared stage: attend to shared (prompt) KV =====
  const int64_t unshared_offset =
      static_cast<int64_t>(FLAGS_max_tokens_per_batch);
  const int64_t shared_kv_capacity =
      std::min(attn_metadata.full_k_cache.size(0), unshared_offset);

  torch::Tensor shared_k_cache =
      attn_metadata.full_k_cache.slice(0, 0, shared_kv_capacity);
  torch::Tensor shared_v_cache =
      attn_metadata.full_v_cache.slice(0, 0, shared_kv_capacity);

  AttentionMetadata shared_attn_meta = attn_metadata;

  shared_attn_meta.plan_info = attn_metadata.shared_plan_info;
  shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;
  VLOG(kGraphExecutorLogVerboseLevel)
      << "shared_attn_meta.q_cu_seq_lens:" << shared_attn_meta.q_cu_seq_lens;
  shared_attn_meta.is_causal = false;

  auto& workspace = flashinfer::FlashinferWorkspace::get_instance();
  torch::Tensor shared_int_workspace_buffer =
      workspace.get_int_workspace_buffer();
  torch::Tensor shared_page_locked_int_workspace_buffer =
      workspace.get_page_locked_int_workspace_buffer();
  CHECK(shared_int_workspace_buffer.defined() &&
        shared_page_locked_int_workspace_buffer.defined())
      << "workspace buffers must be initialized.";

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.shared_plan_info->plan_info.defined())
        << "shared stage plan_info should not be null when enable_cuda_graph "
           "is true";
  } else {
    // Update shared stage plan info (ragged prefill, non-causal).
    flashinfer::update_plan_info(
        attn_metadata.shared_plan_info,
        xllm::kernel::cuda::determine_attention_backend(
            /*pos_encoding_mode=*/0,
            /*use_fp16_qk_reduction=*/false,
            /*use_custom_mask=*/false,
            /*causal=*/false),
        shared_attn_meta,
        query.scalar_type(),
        shared_k_cache.scalar_type(),
        cache.shared_o.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size*/ 1,
        /*window_size_left*/ -1,
        /*enable_cuda_graph*/ false,
        /*causal*/ false,
        /*use_tensor_core*/ true,
        /*force_prefill_plan*/ true);
  }

  xllm::kernel::AttentionParams shared_attention_params(shared_attn_meta);
  shared_attention_params.return_lse = true;
  shared_attention_params.query = query;
  shared_attention_params.output = cache.shared_o;
  shared_attention_params.output_lse = cache.shared_lse;
  shared_attention_params.window_size_left = -1;
  shared_attention_params.scale = scale_;
  shared_attention_params.float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  shared_attention_params.int_workspace_buffer = shared_int_workspace_buffer;
  shared_attention_params.page_locked_int_workspace_buffer =
      shared_page_locked_int_workspace_buffer;
  shared_attention_params.key = shared_k_cache;
  shared_attention_params.value = shared_v_cache;

  xllm::kernel::batch_prefill(shared_attention_params);

  // ===== Unshared stage: attend to unshared (per-beam) KV =====
  AttentionMetadata unshared_attn_meta = attn_metadata;
  unshared_attn_meta.plan_info = attn_metadata.unshared_plan_info;
  unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
  unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
  unshared_attn_meta.paged_kv_last_page_len =
      cache.paged_kv_last_page_len_expanded;

  torch::Tensor unshared_int_workspace_buffer =
      workspace.get_two_stage_unshared_int_workspace_buffer();
  torch::Tensor unshared_page_locked_int_workspace_buffer =
      workspace.get_two_stage_unshared_page_locked_int_workspace_buffer();
  CHECK(unshared_int_workspace_buffer.defined() &&
        unshared_page_locked_int_workspace_buffer.defined())
      << "two-stage unshared workspace buffers must be initialized.";

  const int64_t max_decode_step = attn_metadata.unshared_k_cache.size(2);
  torch::Tensor unshared_k_cache = attn_metadata.unshared_k_cache.view(
      {total_beam, max_decode_step, num_kv_heads_, head_size_});
  torch::Tensor unshared_v_cache = attn_metadata.unshared_v_cache.view(
      {total_beam, max_decode_step, num_kv_heads_, head_size_});

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.unshared_plan_info->plan_info.defined())
        << "unshared stage plan_info should not be null when enable_cuda_graph "
           "is true";
  } else {
    flashinfer::update_plan_info(attn_metadata.unshared_plan_info,
                                 /*backend*/ "fa3",
                                 unshared_attn_meta,
                                 query.scalar_type(),
                                 unshared_k_cache.scalar_type(),
                                 cache.unshared_o.scalar_type(),
                                 head_size_,
                                 head_size_,
                                 num_heads_,
                                 num_kv_heads_,
                                 /*block_size*/ max_decode_step,
                                 /*window_size_left*/ sliding_window_,
                                 /*enable_cuda_graph*/ false,
                                 /*causal*/ false,
                                 /*use_tensor_core*/ false,
                                 /*force_prefill_plan*/ false);
  }

  xllm::kernel::AttentionParams unshared_attention_params(unshared_attn_meta);
  unshared_attention_params.return_lse = true;
  unshared_attention_params.query = query;
  unshared_attention_params.output = cache.unshared_o;
  unshared_attention_params.output_lse = cache.unshared_lse;
  unshared_attention_params.window_size_left = sliding_window_;
  unshared_attention_params.scale = scale_;
  unshared_attention_params.float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  unshared_attention_params.int_workspace_buffer =
      unshared_int_workspace_buffer;
  unshared_attention_params.page_locked_int_workspace_buffer =
      unshared_page_locked_int_workspace_buffer;
  unshared_attention_params.k_cache = unshared_k_cache;
  unshared_attention_params.v_cache = unshared_v_cache;
  unshared_attention_params.use_tensor_core = false;

  xllm::kernel::batch_decode(unshared_attention_params);

  // ===== Combine shared/unshared results =====
  xllm::kernel::cuda::lse_combine(output,
                                  cache.shared_o,
                                  cache.shared_lse,
                                  cache.unshared_o,
                                  cache.unshared_lse);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> XAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
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

  if (attn_metadata.is_prefill) {
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";

    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        xllm::kernel::cuda::determine_attention_backend(
            /*pos_encoding_mode=*/0,
            /*use_fp16_qk_reduction=*/false,
            /*use_custom_mask=*/false,
            /*causal=*/attn_metadata.is_causal),
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size*/ k_cache.size(1),
        /*window_size_left*/ sliding_window_,
        /*enable_cuda_graph*/ false,
        /*causal*/ attn_metadata.is_causal,
        /*use_tensor_core*/ true);

    xllm::kernel::cuda::prefill_reshape_and_cache(
        key, value, attn_metadata.full_k_cache, attn_metadata.full_v_cache);
    xllm::kernel::AttentionParams attention_params(attn_metadata);
    attention_params.query = query;
    attention_params.output = output;
    attention_params.output_lse = output_lse;
    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    // for flashinfer
    attention_params.float_workspace_buffer =
        flashinfer::FlashinferWorkspace::get_instance()
            .get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        flashinfer::FlashinferWorkspace::get_instance()
            .get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        flashinfer::FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();

    attention_params.key = key;
    attention_params.value = value;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    key = key.contiguous();
    value = value.contiguous();

    xllm::kernel::cuda::decoder_reshape_and_cache(
        key,
        value,
        attn_metadata.unshared_k_cache,
        attn_metadata.unshared_v_cache,
        attn_metadata.block_table,
        attn_metadata.step_tensor);

    const bool enable_two_stage_decode =
        FLAGS_enable_xattention_two_stage_decode;

    if (enable_two_stage_decode) {
      run_two_stage_decode(attn_metadata, query, output);
    } else {
      run_single_stage_decode(attn_metadata, key, query, output);
    }
  }
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
