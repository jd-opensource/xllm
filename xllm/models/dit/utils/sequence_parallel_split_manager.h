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

#include <torch/torch.h>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {

// Sequence Parallel helpers (Ulysses-style)
inline torch::Tensor gather_sequence(const torch::Tensor& input_,
                                     int64_t dim,
                                     ProcessGroup* pg) {
  int64_t group_size = pg->world_size();
  auto input = input_.contiguous();
  if (group_size == 1) {
    return input;
  }
  auto tensor_list = parallel_state::gather(input, pg, dim);
  return torch::cat(tensor_list, dim);
}

inline torch::Tensor split_sequence(const torch::Tensor& input,
                                    int64_t dim,
                                    ProcessGroup* pg) {
  int64_t group_size = pg->world_size();
  int64_t rank = pg->rank();
  if (group_size == 1) {
    return input;
  }
  torch::Tensor input_ = input;
  int64_t dim_size = input_.size(dim);
  auto tensor_list = torch::split(input_, dim_size / group_size, dim);
  return tensor_list[rank].contiguous();
}

}  // namespace xllm
