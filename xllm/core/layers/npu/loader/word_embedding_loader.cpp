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
#include "word_embedding_loader.h"

namespace xllm {
namespace layer {

WordEmbeddingLoader::WordEmbeddingLoader(uint64_t weight_count,
                                         const ModelContext& context,
                                         LoadMode mode)
    : BaseLoader(weight_count, context, mode) {}

void WordEmbeddingLoader::load_state_dict(const StateDict& state_dict) {
  if (dp_size_ > 1 || cp_size_ > 1) {
    set_weight(state_dict,
               "weight",
               0,
               1,
               dp_local_tp_rank_,
               dp_local_tp_size_,
               load_to_host());
  } else {
    set_weight(state_dict, "weight", 0, 1, load_to_host());
  }
}

void WordEmbeddingLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  CHECK(working_tensors()[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace xllm
