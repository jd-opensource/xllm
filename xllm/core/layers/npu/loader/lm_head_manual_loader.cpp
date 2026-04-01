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

#include "lm_head_manual_loader.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor prune_lm_head_weight(const torch::Tensor& tensor,
                                   const std::vector<int64_t>& token_ids) {
  if (token_ids.empty()) {
    return tensor;
  }

  CHECK_GE(tensor.dim(), 1)
      << "lm_head weight tensor must have at least 1 dimension";
  for (auto token_id : token_ids) {
    CHECK_GE(token_id, 0) << "candidate token id " << token_id
                          << " is negative";
    CHECK_LT(token_id, tensor.size(0))
        << "candidate token id " << token_id << " exceeds lm_head rows "
        << tensor.size(0);
  }

  auto indices = torch::tensor(
      token_ids,
      torch::TensorOptions().dtype(torch::kInt64).device(tensor.device()));
  return tensor.index_select(/*dim=*/0, indices);
}

}  // namespace

LmHeadManualLoader::LmHeadManualLoader(uint64_t weight_count,
                                       const ModelContext& context)
    : BaseManualLoader(weight_count, context),
      candidate_token_ids_(context.get_model_args().candidate_token_ids()) {
  auto options = context.get_tensor_options();
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void LmHeadManualLoader::load_state_dict(const StateDict& state_dict) {
  if (!state_dict.get_tensor("weight").defined()) {
    return;
  }
  if (cp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 0, dp_local_tp_rank_, dp_local_tp_size_, true);
  } else if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 1, dp_local_tp_rank_, dp_local_tp_size_, true);
  } else {
    set_weight(state_dict, "weight", 0, 1, true);
  }
  at_host_weight_tensors_[0] =
      prune_lm_head_weight(at_host_weight_tensors_[0], candidate_token_ids_);
}

void LmHeadManualLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  CHECK(at_host_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final lm_head weight is not loaded for " << weight_str;
}

void LmHeadManualLoader::merge_host_at_weights() {}

}  // namespace layer
}  // namespace xllm
