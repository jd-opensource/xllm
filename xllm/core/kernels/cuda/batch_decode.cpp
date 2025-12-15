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

#include <unordered_map>

#include "cuda_ops_api.h"
#include "function_factory.h"
#include "util/utils.h"

namespace {
// torch tensor is only on cpu
std::unordered_map<std::string, torch::Tensor> cache_buffer_map;

torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  int32_t seq_len_pow2 = xllm::util::ceil_pow2(seq_len);

  std::string key = std::string("range_") + std::to_string(seq_len_pow2);
  auto it = cache_buffer_map.find(key);
  if (it != cache_buffer_map.end()) {
    return it->second.slice(0, 0, seq_len);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor buffer = torch::arange(seq_len_pow2, options);
  cache_buffer_map.insert(std::make_pair(key, buffer));
  return buffer.slice(0, 0, seq_len);
}
}  // namespace

namespace xllm::kernel::cuda {

void batch_decode(torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool enable_cuda_graph,
                  bool use_tensor_core,
                  torch::Tensor kv_seq_lens) {
  if (use_tensor_core) {
    std::string uri = get_batch_prefill_uri(/*backend=*/"fa2",
                                            query.scalar_type(),
                                            k_cache.scalar_type(),
                                            output.scalar_type(),
                                            paged_kv_indptr.scalar_type(),
                                            query.size(-1),
                                            v_cache.size(-1),
                                            /*pos_encoding_mode=*/0,
                                            /*use_sliding_window=*/false,
                                            /*use_logits_soft_cap=*/false,
                                            /*use_fp16_qk_reduction=*/false);

    const int64_t batch_size = paged_kv_last_page_len.size(0);
    torch::Tensor qo_indptr_host =
        get_cache_buffer(batch_size + 1, torch::kCPU);
    torch::Tensor qo_indptr = qo_indptr_host.to(torch::kCUDA);
    torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
    torch::Tensor kv_len_arr_host = kv_seq_lens.to(torch::kCPU);

    auto plan_info =
        FunctionFactory::get_instance().fa2_prefill_plan_func(uri).call(
            float_workspace_buffer,
            int_workspace_buffer,
            page_locked_int_workspace_buffer,
            qo_indptr_host,
            paged_kv_indptr_host,
            kv_len_arr_host,
            batch_size,  // total_num_rows
            batch_size,
            query.size(1),    // num_qo_heads
            k_cache.size(2),  // num_kv_heads
            k_cache.size(1),  // block_size
            enable_cuda_graph,
            query.size(-1),    // head_dim_qk
            v_cache.size(-1),  // head_dim_vo
            /*causal=*/false);

    FunctionFactory::get_instance().fa2_prefill_paged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        k_cache,
        v_cache,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        output,
        output_lse,
        /*mask_mode_code=*/0,  // NON_CAUSAL
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
  } else {
    std::string uri = get_batch_decode_uri(query.scalar_type(),
                                           k_cache.scalar_type(),
                                           output.scalar_type(),
                                           paged_kv_indptr.scalar_type(),
                                           query.size(-1),
                                           v_cache.size(-1),
                                           /*pos_encoding_mode=*/0,
                                           /*use_sliding_window=*/false,
                                           /*use_logits_soft_cap=*/false);

    torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = paged_kv_last_page_len.size(0);

    torch::Tensor empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query.scalar_type()));
    torch::Tensor empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(k_cache.scalar_type()));

    auto plan_info = FunctionFactory::get_instance().decode_plan_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        page_locked_int_workspace_buffer,
        paged_kv_indptr_host,
        batch_size,
        query.size(1),    // num_qo_heads
        k_cache.size(2),  // num_kv_heads
        k_cache.size(1),  // block_size
        enable_cuda_graph,
        window_left,
        /*logits_soft_cap=*/0.0,
        query.size(-1),    // head_dim_qk
        v_cache.size(-1),  // head_dim_vo
        empty_q_data,
        empty_kv_data);

    FunctionFactory::get_instance().decode_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        k_cache,
        v_cache,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        output,
        output_lse,
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0);
  }
}

}  // namespace xllm::kernel::cuda
