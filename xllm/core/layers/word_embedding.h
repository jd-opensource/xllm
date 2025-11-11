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

#if defined(USE_NPU)
#include "npu/npu_word_embedding_impl.h"
#endif
#include "common/word_embedding_impl.h"

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class NpuWordEmbedding : public torch::nn::ModuleHolder<NpuWordEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<NpuWordEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuWordEmbeddingImpl;
  NpuWordEmbedding(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuWordEmbeddingImpl>(context)) {}
};

#endif

class WordEmbedding : public torch::nn::ModuleHolder<WordEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<WordEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = WordEmbeddingImpl;
  WordEmbedding(int64_t num_embeddings,
                int64_t embedding_dim,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<WordEmbeddingImpl>(num_embeddings,
                                                         embedding_dim,
                                                         parallel_args,
                                                         options)) {}
};

}  // namespace layer
}  // namespace xllm
