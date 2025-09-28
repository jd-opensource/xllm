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

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "options.h"

namespace xllm {

class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;

  virtual ForwardInput prepare_inputs(Batch& batch) = 0;

  virtual torch::Tensor run(const std::vector<torch::Tensor>& tokens,
                            const std::vector<torch::Tensor>& positions,
                            std::vector<KVCache>& kv_caches,
                            const std::vector<ModelInputParams>& params) = 0;
};

}  // namespace xllm
