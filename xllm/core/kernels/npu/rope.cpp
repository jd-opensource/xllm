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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& positions) {
  // FIXME: This computation of 'cos' and 'sin' should only be performed
  // for the first layer (or if the cache is empty). For subsequent layers,
  // the calculated 'cos' and 'sin' values from the first layer should be
  // reused/cached to avoid redundant computation.
  auto cos_sin = cos_sin_cache.index_select(0, positions);
  int64_t last_dim = cos_sin.size(-1);
  auto cos_sin_vec = cos_sin.view({-1, 2, last_dim / 2})
                         .repeat({1, 1, 2})
                         .chunk(2, /*dim=*/-2);
  auto cos = cos_sin_vec[0].view({1, -1, 1, last_dim});
  auto sin = cos_sin_vec[1].view({1, -1, 1, last_dim});

  const int64_t rotary_dim = sin.size(-1);
  q = q.view({1, q.size(0), -1, rotary_dim});
  k = k.view({1, k.size(0), -1, rotary_dim});

  at_npu::native::custom_ops::npu_apply_rotary_pos_emb(q, k, cos, sin);
}

}  // namespace xllm::kernel::npu