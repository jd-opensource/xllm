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

#include "rms_norm_loader.h"

namespace xllm {
namespace layer {

RMSNORMLoader::RMSNORMLoader(uint64_t weight_count, const ModelContext& context)
    : BaseLoader(weight_count, context) {
  at_weight_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void RMSNORMLoader::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
  at_weight_tensors_[0] = at_weight_tensors_[0].to(dtype_);
}

void RMSNORMLoader::verify_loaded_weights(const std::string& weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace xllm