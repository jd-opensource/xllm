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

#include "attention_metadata.h"

#include "core/common/global_flags.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"

namespace xllm {
namespace layer {

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params) {
  return AttentionMetadata::build(params, "float");
}

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           const std::string& compute_dtype) {
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

  attn_metadata.is_chunked_prefill =
      params.batch_forward_type.is_chunked_prefill();
  attn_metadata.is_prefill = params.batch_forward_type.is_prefill();
  if (!attn_metadata.is_prefill || FLAGS_enable_mla) {
    attn_metadata.block_table = params.block_tables;
    attn_metadata.kv_seq_lens = torch::diff(params.kv_seq_lens);  // kv seqlens
  }

  attn_metadata.is_dummy = (params.q_max_seq_len == 0);

  return attn_metadata;
}

void AttentionMetadata::update(c10::ScalarType query_dtype,
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
  // for flashinfer
  // TODO: check if not flashinfer backend, we return.
  if (layer_id != 0) return;

  // we ready flash_planinfo and flashinfer_uri in the first step
  if (causal) {
    uri = kernel::cuda::get_batch_prefill_uri(
        /*backend=*/"fa2",
        query_dtype,
        key_dtype,
        output_dtype,
        q_cu_seq_lens.scalar_type(),
        head_dim_qk,
        head_dim_vo,
        /*pos_encoding_mode=*/0,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_fp16_qk_reduction=*/false);

    torch::Tensor qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_cu_seq_lens_host = kv_cu_seq_lens.to(torch::kCPU);
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
    uri = kernel::cuda::get_batch_decode_uri(query_dtype,
                                             key_dtype,
                                             output_dtype,
                                             paged_kv_indptr.scalar_type(),
                                             head_dim_qk,
                                             head_dim_vo,
                                             /*pos_encoding_mode=*/0,
                                             /*use_sliding_window=*/false,
                                             /*use_logits_soft_cap=*/false);

    torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = paged_kv_last_page_len.size(0);
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
