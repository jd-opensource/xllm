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

namespace xllm::mlu {

void apply_rotary(const torch::Tensor& input,
                  torch::Tensor& output,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const std::optional<torch::Tensor>& position_ids,
                  const torch::Tensor& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk,
                  int max_query_len) {
  tmo::torch_api::apply_rotary(input,
                               output,
                               sin,
                               cos,
                               position_ids,
                               cu_query_lens,
                               interleaved,
                               discrete,
                               dynamic_ntk,
                               max_query_len);
}

}  // namespace xllm::mlu