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

#include "npu_cp_ep_padding.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "common/global_flags.h"
#include "util/tensor_helper.h"

namespace xllm {

void CpEpPaddingData::set_placeholder(const torch::Tensor& placeholder) {
  attn_padding_idx_ = placeholder;
  attn_unpadding_idx_ = placeholder;
  ffn_padding_idx_ = placeholder;
  ffn_unpadding_idx_ = placeholder;
  lm_head_skip_padding_token_indices_ = placeholder;
  gather_prenorm_idx_ = placeholder;
}

CpEpPadding::CpEpPadding(const torch::Tensor& input_ids,
                         int32_t num_experts_per_tok,
                         const nlohmann::json& mapping_npu,
                         at::Device device,
                         torch::ScalarType dtype,
                         bool is_prefill)
    : device_(device) {
  (void)num_experts_per_tok;
  (void)dtype;

  attn_tp_size_ = mapping_npu["attnTpSize"].get<int64_t>();
  attn_tp_rank_ = mapping_npu["attnTp"]["rank"].get<int64_t>();
  attn_cp_size_ = mapping_npu["attnCp"]["rankIds"].size();
  input_length_ = std::max<int64_t>(input_ids.numel(), 1);

  is_dynamic_ep_ = FLAGS_expert_parallel_degree == 2 ||
                   (FLAGS_expert_parallel_degree == 3 && is_prefill);
}

CpEpPaddingData CpEpPadding::build() {
  prepare_indices();
  return assemble_result();
}

void CpEpPadding::prepare_indices() {
  CHECK_GT(attn_tp_size_, 0) << "attnTpSize must be positive.";
  CHECK_GT(attn_cp_size_, 0) << "attnCp group size must be positive.";
  CHECK_LT(attn_tp_rank_, attn_tp_size_)
      << "attnTp rank must be smaller than attnTpSize.";

  const int64_t padding_length =
      (attn_tp_size_ - input_length_ % attn_tp_size_) % attn_tp_size_;
  const int64_t input_len_padding_per_group = input_length_ + padding_length;
  const int64_t input_len_padding_per_rank =
      input_len_padding_per_group / attn_tp_size_;

  attn_padding_idx_ =
      torch::cat({torch::arange(input_length_, torch::kInt32),
                  torch::zeros({padding_length}, torch::kInt32)});

  gather_prenorm_idx_ =
      attn_padding_idx_.slice(0,
                              attn_tp_rank_ * input_len_padding_per_rank,
                              (attn_tp_rank_ + 1) * input_len_padding_per_rank);

  std::vector<torch::Tensor> all_gather_skip_components;
  all_gather_skip_components.reserve(attn_cp_size_);
  for (int64_t cp_rank = 0; cp_rank < attn_cp_size_; ++cp_rank) {
    all_gather_skip_components.emplace_back(
        torch::arange(input_length_, torch::kInt32) +
        cp_rank * input_len_padding_per_group);
  }
  const auto all_gather_skip_padding_token_indices =
      torch::cat(all_gather_skip_components, 0);

  if (is_dynamic_ep_) {
    attn_unpadding_idx_ =
        torch::arange(input_len_padding_per_rank, torch::kInt32);
    ffn_padding_idx_ = attn_unpadding_idx_;
  } else {
    attn_unpadding_idx_ = all_gather_skip_padding_token_indices;

    std::vector<torch::Tensor> ffn_padding_components;
    ffn_padding_components.reserve(attn_cp_size_);
    for (int64_t cp_rank = 0; cp_rank < attn_cp_size_; ++cp_rank) {
      ffn_padding_components.emplace_back(
          torch::cat({torch::arange(input_length_ * cp_rank,
                                    input_length_ * (cp_rank + 1),
                                    torch::kInt32),
                      torch::zeros({padding_length}, torch::kInt32)}));
    }
    ffn_padding_idx_ = torch::cat(ffn_padding_components, 0);
  }

  ffn_unpadding_idx_ = torch::arange(input_length_, torch::kInt32);
  lm_head_skip_padding_token_indices_ = all_gather_skip_padding_token_indices;
}

CpEpPaddingData CpEpPadding::assemble_result() const {
  CpEpPaddingData result;
  result.attn_padding_idx(safe_to(attn_padding_idx_, device_, true))
      .attn_unpadding_idx(safe_to(attn_unpadding_idx_, device_, true))
      .ffn_padding_idx(safe_to(ffn_padding_idx_, device_, true))
      .ffn_unpadding_idx(safe_to(ffn_unpadding_idx_, device_, true))
      .lm_head_skip_padding_token_indices(
          safe_to(lm_head_skip_padding_token_indices_, device_, true))
      .gather_prenorm_idx(safe_to(gather_prenorm_idx_, device_, true));
  return result;
}

}  // namespace xllm
