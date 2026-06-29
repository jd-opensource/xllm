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

#include "causal_conv1d_update_decode.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <algorithm>
#include <unordered_map>

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/mlu/utils.h"

namespace xllm {
namespace kernel {
namespace mlu {

torch::Tensor causal_conv1d_update_decode(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::Tensor& conv_state_indices,
    int32_t pad_slot_id,
    const std::optional<torch::Tensor>& query_start_loc_opt,
    int32_t max_query_len,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt) {
  bool unsqueeze = (x.dim() == 2);
  torch::Tensor x_input = x;
  if (unsqueeze) {
    x_input = x.unsqueeze(-1);
  }

  int32_t batch = static_cast<int32_t>(conv_state_indices.size(0));
  int32_t dim = static_cast<int32_t>(x_input.size(1));
  int32_t seqlen = static_cast<int32_t>(x_input.size(2));
  int32_t width = static_cast<int32_t>(weight.size(1));
  int32_t state_len = width - 1;
  int32_t num_cache_lines = static_cast<int32_t>(conv_state.size(0));

  // Create output tensor with same shape and dtype as x
  torch::Tensor out = torch::empty_like(x_input);

  // Strides for x (batch, dim, seqlen)
  int32_t stride_x_seq = static_cast<int32_t>(x_input.stride(0));
  int32_t stride_x_dim = static_cast<int32_t>(x_input.stride(1));
  int32_t stride_x_token = static_cast<int32_t>(x_input.stride(2));
  // Strides for weight (dim, width)
  int32_t stride_w_dim = static_cast<int32_t>(weight.stride(0));
  int32_t stride_w_width = static_cast<int32_t>(weight.stride(1));
  // Strides for conv_state (num_cache_lines, dim, state_len)
  int32_t stride_istate_seq = static_cast<int32_t>(conv_state.stride(0));
  int32_t stride_istate_dim = static_cast<int32_t>(conv_state.stride(1));
  int32_t stride_istate_tok = static_cast<int32_t>(conv_state.stride(2));
  // Strides for conv_state_indices (batch)
  int32_t stride_state_indices =
      static_cast<int32_t>(conv_state_indices.stride(0));
  // Strides for out (batch, dim, seqlen)
  int32_t stride_o_seq = static_cast<int32_t>(out.stride(0));
  int32_t stride_o_dim = static_cast<int32_t>(out.stride(1));
  int32_t stride_o_token = static_cast<int32_t>(out.stride(2));

  // Data pointers
  void* x_ptr = x_input.data_ptr();
  void* weight_ptr = weight.data_ptr();
  void* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr() : nullptr;
  void* conv_state_ptr = conv_state.data_ptr();
  void* conv_state_indices_ptr = conv_state_indices.data_ptr();
  void* num_accepted_tokens_ptr = num_accepted_tokens_opt.has_value()
                                      ? num_accepted_tokens_opt->data_ptr()
                                      : nullptr;
  void* query_start_loc_ptr = query_start_loc_opt.has_value()
                                  ? query_start_loc_opt->data_ptr()
                                  : nullptr;
  void* block_idx_last_scheduled_token_ptr =
      block_idx_last_scheduled_token_opt.has_value()
          ? block_idx_last_scheduled_token_opt->data_ptr()
          : nullptr;
  void* initial_state_idx_ptr = initial_state_idx_opt.has_value()
                                    ? initial_state_idx_opt->data_ptr()
                                    : nullptr;
  void* out_ptr = out.data_ptr();

  bool has_bias = bias_opt.has_value();

  constexpr int32_t kBd = 8;
  int32_t num_feature_blocks = (dim + kBd - 1) / kBd;
  cnrtDim3_t dim_block = {static_cast<uint32_t>(num_feature_blocks),
                          static_cast<uint32_t>(batch),
                          1};

  auto queue = torch_mlu::getCurMLUStream();

  // algo_id: select pre-compiled kernel variant based on dim value
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
  int32_t algo_id = lookup_algo_id(kDimToAlgoId, dim, /*dim_name=*/"dim");

  tmo_causal_conv1d_update_decode_kernel(queue,
                                         &dim_block,
                                         x_ptr,
                                         weight_ptr,
                                         bias_ptr,
                                         conv_state_ptr,
                                         conv_state_indices_ptr,
                                         num_accepted_tokens_ptr,
                                         query_start_loc_ptr,
                                         block_idx_last_scheduled_token_ptr,
                                         initial_state_idx_ptr,
                                         out_ptr,
                                         batch,
                                         num_cache_lines,
                                         stride_x_seq,
                                         stride_x_dim,
                                         stride_x_token,
                                         stride_w_dim,
                                         stride_w_width,
                                         stride_istate_seq,
                                         stride_istate_dim,
                                         stride_istate_tok,
                                         stride_state_indices,
                                         stride_o_seq,
                                         stride_o_dim,
                                         stride_o_token,
                                         pad_slot_id,
                                         algo_id);

  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out;
}

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
