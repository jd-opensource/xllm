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

#include "layer_utils.h"

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {

void update_dummy_run_input(int64_t dp_rank,
                            torch::Tensor& positions,
                            ModelInputParams& input_params) {
  auto& dp_ranks = input_params.dp_global_token_nums;
  bool is_dummy_run = dp_ranks[dp_rank] == 0;
  for (size_t i = 0; i < dp_ranks.size(); i++) {
    if (dp_ranks[i] == 0) {
      dp_ranks[i] = 1;
    }
  }
  if (is_dummy_run) {
    positions = torch::tensor({1}).to(torch::kInt32).to(positions.device());
  }
}

}  // namespace layer
}  // namespace xllm
