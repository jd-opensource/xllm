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

#include "word_embedding.h"

#include "atb_word_embedding_impl.h"

namespace xllm::hf {

std::shared_ptr<AtbEmbeddingImpl> create_word_embedding_layer(
    const Context& context) {
  return std::make_shared<AtbWordEmbeddingImpl>(context);
}

AtbWordEmbedding::AtbWordEmbedding(const Context& context)
    : ModuleHolder(create_word_embedding_layer(context)) {}

}  // namespace xllm::hf
