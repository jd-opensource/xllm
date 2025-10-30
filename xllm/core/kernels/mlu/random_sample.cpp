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

namespace xllm::kernel::mlu {

torch::Tensor random_sample(const torch::Tensor& probs) {
  torch::Tensor flat_probs;
  if (probs.dim() == 3) {
    flat_probs = probs.reshape({-1, probs.size(2)});
  } else {
    flat_probs = probs;
  }
  auto output =
      torch::empty({flat_probs.size(0), 1},
                   torch::dtype(torch::kInt64).device(probs.device()));
  tmo::torch_api::random_sample(flat_probs, output, true, torch::Generator());
  if (probs.dim() == 3) {
    return output.reshape({probs.size(0), probs.size(1)});
  }
  return output.flatten();
}

}  // namespace xllm::kernel::mlu