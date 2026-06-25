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

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "causal_conv1d_fwd_vllm_kernel.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/mlu/utils.h"

namespace xllm {
namespace kernel {
namespace mlu {
torch::Tensor causal_conv1d_fn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& batch,
    const torch::Tensor& token_block_offset,
    int32_t nt,
    const std::optional<torch::Tensor>& bias_opt,
    const std::optional<torch::Tensor>& cache_indices_opt,
    const std::optional<torch::Tensor>& has_initial_state_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool inplace_final_state) {
  auto out = torch::zeros_like(x);
  int32_t cu_seqlen = x.size(1);
  int32_t num_cache_lines = conv_states.size(0);
  int32_t stride_x_dim = static_cast<int32_t>(x.stride(0));
  int32_t stride_x_token = static_cast<int32_t>(x.stride(1));
  int32_t stride_w_dim = static_cast<int32_t>(weight.stride(0));
  int32_t stride_w_width = static_cast<int32_t>(weight.stride(1));
  int32_t stride_istate_seq = static_cast<int32_t>(conv_states.stride(0));
  int32_t stride_istate_dim = static_cast<int32_t>(conv_states.stride(1));
  int32_t stride_istate_token = static_cast<int32_t>(conv_states.stride(2));
  int32_t stride_cache_indices =
      cache_indices_opt.has_value()
          ? static_cast<int32_t>(cache_indices_opt.value().stride(0))
          : 0;
  int32_t stride_o_dim, stride_o_token;
  if (out.dim() == 2) {
    stride_o_dim = static_cast<int32_t>(out.stride(0));
    stride_o_token = static_cast<int32_t>(out.stride(1));
  } else {
    stride_o_dim = static_cast<int32_t>(out.stride(1));
    stride_o_token = static_cast<int32_t>(out.stride(2));
  }

  void* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr() : nullptr;
  void* x_ptr = x.data_ptr();
  void* weight_ptr = weight.data_ptr();
  void* conv_states_ptr = conv_states.data_ptr();
  void* cache_indices_ptr = cache_indices_opt.has_value()
                                ? cache_indices_opt.value().data_ptr()
                                : nullptr;
  void* has_initial_state_ptr = has_initial_state_opt.has_value()
                                    ? has_initial_state_opt.value().data_ptr()
                                    : nullptr;
  void* query_start_loc_ptr = query_start_loc.data_ptr();
  void* batch_ptr = batch.data_ptr();
  void* token_block_offset_ptr = token_block_offset.data_ptr();
  void* out_ptr = out.data_ptr();

  // grid
  int32_t num_programs =
      static_cast<int32_t>(std::min(nt, static_cast<int32_t>(8)));
  cnrtDim3_t dim_block = {static_cast<uint32_t>(num_programs), 1, 1};
  auto queue = torch_mlu::getCurMLUStream();

  static const std::unordered_map<int32_t, int32_t> kDimToAlgoId = {
      {384, 0},
      {512, 1},
      {640, 2},
      {768, 3},
      {1024, 4},
      {1280, 5},
      {1536, 6},
      {2048, 7},
      {2560, 8},
      {3072, 9},
      {4096, 10},
      {5120, 11},
      {6144, 12},
      {8192, 13},
      {10240, 14},
      {12288, 15},
  };
  int32_t algo_id = lookup_algo_id(kDimToAlgoId,
                                   static_cast<int32_t>(x.size(0)),
                                   /*dim_name=*/"dim");
  tmo_causal_conv1d_fwd_vllm_kernel(queue,
                                    &dim_block,
                                    x_ptr,
                                    weight_ptr,
                                    bias_ptr,
                                    conv_states_ptr,
                                    cache_indices_ptr,
                                    has_initial_state_ptr,
                                    query_start_loc_ptr,
                                    batch_ptr,
                                    token_block_offset_ptr,
                                    nullptr,  // block_idx_first_scheduled_token
                                    nullptr,  // block_idx_last_scheduled_token
                                    nullptr,  // initial_state_idx
                                    nullptr,  // num_computed_tokens
                                    out_ptr,
                                    nt,
                                    cu_seqlen,
                                    num_cache_lines,
                                    stride_x_dim,
                                    stride_x_token,
                                    stride_w_dim,
                                    stride_w_width,
                                    stride_istate_seq,
                                    stride_istate_dim,
                                    stride_istate_token,
                                    stride_cache_indices,
                                    stride_o_dim,
                                    stride_o_token,
                                    algo_id);
  return out;
}
}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
