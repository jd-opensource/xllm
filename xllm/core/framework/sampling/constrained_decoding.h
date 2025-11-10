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
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace xllm {

// constrained decoding is used to ensure that the generated content
// conforms to specific formats or rules.
class ConstrainedDecoding {
 public:
  virtual ~ConstrainedDecoding();

  virtual bool build_mask_cache();

  // input generated_token_list: [sequence_num][generated_token_ids]
  // output: mask tensor[sequence_num,vocab_size]
  virtual torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list);
};
}  // namespace xllm
