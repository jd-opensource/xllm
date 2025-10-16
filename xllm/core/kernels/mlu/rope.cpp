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

#include "mlu_ops_api.h"
#include "torch_mlu_ops.h"

namespace xllm::kernel::mlu {

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const std::optional<torch::Tensor>& position_ids,
                  const torch::Tensor& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk,
                  int max_query_len) {
  const int64_t rotary_dim = sin.size(-1);
  const int64_t T = q.size(0);
  q = q.view({T, -1});
  k = k.view({T, -1});
  auto qk = torch::cat({q, k}, /*dim=*/-1);
  qk = qk.view({T, -1, rotary_dim});
  tmo::torch_api::apply_rotary(qk,
                               qk /* output */,
                               sin,
                               cos,
                               position_ids,
                               cu_query_lens,
                               interleaved,
                               discrete,
                               false /* dynamic_ntk */,
                               max_query_len);
  qk = qk.view({-1, q.size(-1) + k.size(-1)});
  auto qk_vec = qk.split({q.size(-1), k.size(-1)}, /*dim=*/-1);
  q = qk_vec[0];
  k = qk_vec[1];
}

}  // namespace xllm::kernel::mlu