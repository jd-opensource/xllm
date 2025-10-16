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

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {
namespace layer {

struct AttentionMetadata {
 public:
  static AttentionMetadata build(const ModelInputParams& params,
                                 bool is_prefill);

  static AttentionMetadata build(const ModelInputParams& params,
                                 const std::string& compute_dtype,
                                 bool is_prefill);

  torch::Tensor query_start_loc;
  torch::Tensor seq_start_loc;
  torch::Tensor kv_seq_lens;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
  int max_query_len;
  int max_seq_len;
  std::string compute_dtype;
  bool is_prefill;
  bool is_chunked_prefill;

  // for flashinfer
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
};

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl() = default;

  AttentionImpl(int num_heads,
                int head_size,
                float scale,
                int num_kv_heads,
                int sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

 private:
  int num_heads_;
  int head_size_;
  float scale_;
  int num_kv_heads_;
  int sliding_window_;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace xllm
