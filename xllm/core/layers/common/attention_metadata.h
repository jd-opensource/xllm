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

#pragma once

#include <torch/torch.h>

#include "framework/model/model_input_params.h"

namespace xllm {
namespace layer {

struct AttentionMetadata {
 public:
  static AttentionMetadata build(const ModelInputParams& params);

  static AttentionMetadata build(const ModelInputParams& params,
                                 const std::string& compute_dtype);

  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_seq_lens;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
  int64_t max_query_len;
  int64_t max_seq_len;
  std::string compute_dtype;
  bool is_prefill;
  bool is_chunked_prefill;
  bool is_dummy;

  // for mrope
  torch::Tensor mrope_cos;
  torch::Tensor mrope_sin;

  // for flashinfer
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
};

}  // namespace layer
}  // namespace xllm