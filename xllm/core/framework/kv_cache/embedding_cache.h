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
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/macros.h"
#include "runtime/forward_params.h"

namespace xllm {

class EmbeddingCache final {
 public:
  struct DecodeState {
    // Proposal cached for next decode step.
    torch::Tensor embedding;
    int32_t token_id = -1;             // draft next token for step_decode seed
    int32_t correction_token_id = -1;  // accepted token for step correction
    int32_t correction_position_offset = 0;
    torch::Tensor probs;
  };

  EmbeddingCache(int32_t total_nums);

  ~EmbeddingCache() = default;

  // disable copy, move and assign
  DISALLOW_COPY_AND_ASSIGN(EmbeddingCache);

  void write(const std::vector<int32_t>& embedding_ids,
             const torch::Tensor& next_tokens,
             const torch::Tensor& embeddings,
             const torch::Tensor& probs,
             const torch::Tensor& accepted_tokens = torch::Tensor());

  void set_placeholder(const torch::Tensor& embedding_placeholder);

  ForwardOutput read_for_decode(const std::vector<int32_t>& embedding_ids);
  std::vector<int32_t> read_correction_tokens(
      const std::vector<int32_t>& embedding_ids) const;
  std::vector<int32_t> read_position_offsets(
      const std::vector<int32_t>& embedding_ids) const;

  void clear(const std::vector<int32_t>& embedding_ids);

 private:
  std::vector<DecodeState> decode_tails_;
  torch::Tensor embedding_placeholder_;

  DecodeState& mutable_tail(int32_t embedding_id);
  const DecodeState& get_tail(int32_t embedding_id) const;
};

}  // namespace xllm
