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

#include "attention.h"

#include "flashinfer_workspace.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int layer_id,
                             int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : layer_id_(layer_id),
      num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {
  CHECK(layer_id >= 0) << "layer_id passed to attention is invalid, layer_id = "
                       << layer_id;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  auto output = torch::empty_like(query);
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
  step_wise_attn_state_.update(
      layer_id_,
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
      /*causal*/ attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);

  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key;
  reshape_paged_cache_params.value = value;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  xllm::kernel::AttentionParams attention_params;
  attention_params.query = query;
  attention_params.output = output;
  attention_params.output_lse = output_lse;
  // attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  attention_params.compute_dtype = attn_metadata.compute_dtype;
  // for flashinfer
  attention_params.float_workspace_buffer =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer();
  attention_params.int_workspace_buffer =
      FlashinferWorkspace::get_instance().get_int_workspace_buffer();
  attention_params.page_locked_int_workspace_buffer =
      FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();
  attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  attention_params.uri = step_wise_attn_state_.uri;
  attention_params.plan_info = step_wise_attn_state_.plan_info;

  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";
  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    query = query.view({-1, 1, num_heads_, head_size_});
    output = output.view({-1, 1, num_heads_, head_size_});

    attention_params.query = query;
    attention_params.output = output;
    attention_params.k_cache = k_cache;
    attention_params.v_cache = v_cache;

    // for flashinfer
    attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
    attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
    attention_params.paged_kv_last_page_len =
        attn_metadata.paged_kv_last_page_len;

    xllm::kernel::batch_decode(attention_params);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void StepwiseAttentionState::update(int layer_id,
                                    const AttentionMetadata& attn_meta,
                                    c10::ScalarType query_dtype,
                                    c10::ScalarType key_dtype,
                                    c10::ScalarType output_dtype,
                                    int head_dim_qk,
                                    int head_dim_vo,
                                    int num_qo_heads,
                                    int num_kv_heads,
                                    int block_size,
                                    int window_size_left,
                                    bool enable_cuda_graph,
                                    bool causal) {
  CHECK(layer_id != -1) << "Need to set layer_id to ModelContext or Attention.";

  // for flashinfer
  // TODO: check if not flashinfer backend, we return.
  if (layer_id != 0) return;

  // we ready flash_planinfo and flashinfer_uri in the first step
  if (causal) {
    std::string backend = kernel::cuda::determine_attention_backend(
        /*pos_encoding_mode=*/0,
        /*use_fp16_qk_reduction=*/false,
        /*use_custom_mask=*/false);
    uri = kernel::cuda::get_batch_prefill_uri(
        backend,
        query_dtype,
        key_dtype,
        output_dtype,
        attn_meta.q_cu_seq_lens.scalar_type(),
        head_dim_qk,
        head_dim_vo,
        /*pos_encoding_mode=*/0,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_fp16_qk_reduction=*/false);

    torch::Tensor qo_indptr_host = attn_meta.q_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_cu_seq_lens_host =
        attn_meta.kv_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_len_arr_host =
        kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
    const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
    const int64_t batch_size = qo_indptr_host.size(0) - 1;
    plan_info =
        kernel::cuda::FunctionFactory::get_instance()
            .prefill_plan_func(uri)
            .call(
                FlashinferWorkspace::get_instance()
                    .get_float_workspace_buffer(),
                FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
                FlashinferWorkspace::get_instance()
                    .get_page_locked_int_workspace_buffer(),
                qo_indptr_host,
                kv_cu_seq_lens_host,
                kv_len_arr_host,
                total_num_rows,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                /*page_size=*/1,
                enable_cuda_graph,
                head_dim_qk,
                head_dim_vo,
                causal);
  } else {
    uri = kernel::cuda::get_batch_decode_uri(
        query_dtype,
        key_dtype,
        output_dtype,
        attn_meta.paged_kv_indptr.scalar_type(),
        head_dim_qk,
        head_dim_vo,
        /*pos_encoding_mode=*/0,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false);

    torch::Tensor paged_kv_indptr_host =
        attn_meta.paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
    torch::Tensor empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query_dtype));
    torch::Tensor empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(key_dtype));
    plan_info =
        kernel::cuda::FunctionFactory::get_instance()
            .decode_plan_func(uri)
            .call(
                FlashinferWorkspace::get_instance()
                    .get_float_workspace_buffer(),
                FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
                FlashinferWorkspace::get_instance()
                    .get_page_locked_int_workspace_buffer(),
                paged_kv_indptr_host,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                block_size,
                enable_cuda_graph,
                window_size_left,
                /*logits_soft_cap=*/0.0,
                head_dim_qk,
                head_dim_vo,
                empty_q_data,
                empty_kv_data);
  }
}

}  // namespace layer
}  // namespace xllm
