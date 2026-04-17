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

#include "column_parallel_linear_manual_loader.h"

namespace xllm {
namespace layer {

ColumParallelLinearManualLoader::ColumParallelLinearManualLoader(
    uint64_t weight_count,
    const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  at_host_weight_tensors_[0] = torch::zeros(
      {1}, torch::TensorOptions().dtype(dtype_).device(torch::kCPU));
}

void ColumParallelLinearManualLoader::load_state_dict(
    const StateDict& state_dict) {
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 0, dp_local_tp_rank_, dp_local_tp_size_, true);
  } else {
    set_weight(state_dict, "weight", 0, 0, true);
  }
  at_host_weight_tensors_[0] = at_host_weight_tensors_[0].to(dtype_);
}

void ColumParallelLinearManualLoader::verify_loaded_weights() const {
  verify_loaded_weights("column_parallel_linear");
}

void ColumParallelLinearManualLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  CHECK(at_host_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void ColumParallelLinearManualLoader::merge_host_at_weights() {}

}  // namespace layer
}  // namespace xllm
