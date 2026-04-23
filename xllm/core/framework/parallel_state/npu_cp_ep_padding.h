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

#pragma once

#include <torch/torch.h>

#include "common/macros.h"
#include "util/json_reader.h"

namespace xllm {

struct CpEpPaddingData {
  void set_placeholder(const torch::Tensor& placeholder);

  PROPERTY(torch::Tensor, attn_padding_idx);

  PROPERTY(torch::Tensor, attn_unpadding_idx);

  PROPERTY(torch::Tensor, ffn_padding_idx);

  PROPERTY(torch::Tensor, ffn_unpadding_idx);

  PROPERTY(torch::Tensor, lm_head_skip_padding_token_indices);

  PROPERTY(torch::Tensor, gather_prenorm_idx);
};

class CpEpPadding {
 public:
  CpEpPadding(const torch::Tensor& input_ids,
              int32_t num_experts_per_tok,
              const nlohmann::json& mapping_npu,
              at::Device device,
              torch::ScalarType dtype,
              bool is_prefill);

  CpEpPaddingData build();

 private:
  void prepare_indices();
  CpEpPaddingData assemble_result() const;

  bool is_dynamic_ep_ = false;
  int64_t input_length_ = 0;
  int64_t attn_tp_size_ = 1;
  int64_t attn_tp_rank_ = 0;
  int64_t attn_cp_size_ = 1;

  torch::Tensor attn_padding_idx_;
  torch::Tensor attn_unpadding_idx_;
  torch::Tensor ffn_padding_idx_;
  torch::Tensor ffn_unpadding_idx_;
  torch::Tensor lm_head_skip_padding_token_indices_;
  torch::Tensor gather_prenorm_idx_;

  at::Device device_;
};

}  // namespace xllm
