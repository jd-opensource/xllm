/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

namespace xllm {

DiTForwardInput xllm::DiTBatch::prepare_forward_input() {
  CHECK(!dit_request_vec_.empty());

  DiTForwardInput forward_input;
  forward_input.input_params = dit_request_vec_[0]->state().input_params();
  forward_input.generation_params =
      dit_request_vec_[0]->state().generation_params();

  return forward_input;
}

}  // namespace xllm
