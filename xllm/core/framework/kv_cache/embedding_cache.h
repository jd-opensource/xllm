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
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/macros.h"

namespace xllm {

class EmbeddingCache final {
 public:
  struct DecodeState {
    // Current decode embedding (last accepted token embedding).
    torch::Tensor embedding;
    int32_t last_token_id = -1;
    torch::Tensor prev_embedding;
    int32_t prev_token_id = -1;
    bool need_first_decode_fix = false;
  };

  EmbeddingCache(int32_t total_nums);

  ~EmbeddingCache() = default;

  // disable copy, move and assign
  DISALLOW_COPY_AND_ASSIGN(EmbeddingCache);

  void write(int32_t embedding_id, const torch::Tensor& embeddings);
  void write(const std::vector<int32_t>& embedding_ids,
             const torch::Tensor& embeddings);
  void write_validate(const std::vector<int32_t>& embedding_ids,
                      const torch::Tensor& next_tokens,
                      const torch::Tensor& embeddings,
                      int32_t num_speculative_tokens = -1);

  std::vector<DecodeState> read_for_decode(
      const std::vector<int32_t>& embedding_ids);

  void clear_first_decode_fix(const std::vector<int32_t>& embedding_ids,
                              const std::vector<uint8_t>& clear_mask);

  // Set placeholder tensor for PD separation: when read() finds an empty slot
  // (e.g. first decode on this instance), return placeholder instead so batch
  // can be formed without missing embedding. Shape should be [hidden_size].
  void set_placeholder(const torch::Tensor& placeholder);

  torch::Tensor read(int32_t embedding_id);
  torch::Tensor read(const std::vector<int32_t>& embedding_ids);

 private:
  std::vector<DecodeState> decode_tails_;
  // placeholder for empty slots (e.g. PD separation decode instance)
  torch::Tensor placeholder_;

  DecodeState& mutable_tail(int32_t embedding_id);
  const DecodeState& get_tail(int32_t embedding_id) const;
  void set_last_state(int32_t embedding_id,
                      const torch::Tensor& embedding,
                      int32_t token_id);
};

}  // namespace xllm
