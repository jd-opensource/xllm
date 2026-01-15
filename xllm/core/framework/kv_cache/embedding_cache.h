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
  EmbeddingCache(int32_t total_nums);

  ~EmbeddingCache() = default;

  // disable copy, move and assign
  DISALLOW_COPY_AND_ASSIGN(EmbeddingCache);

  void write(int32_t embedding_id, const torch::Tensor& embeddings);
  void write(const std::vector<int32_t>& embedding_ids,
             const torch::Tensor& embeddings);
  void write_validate(const std::vector<int32_t>& embedding_ids,
                      torch::Tensor& next_tokens,
                      const torch::Tensor& embeddings);

  torch::Tensor read(int32_t embedding_id);
  torch::Tensor read(const std::vector<int32_t>& embedding_ids);

 private:
  // embedding cache
  std::vector<torch::Tensor> cache_;
};

}  // namespace xllm
