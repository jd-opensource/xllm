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

#include "layers/mlu/deepseek_v4/deepseek_v4_topk.h"

#include <glog/logging.h>

#include <tuple>

namespace xllm {
namespace layer {

DeepSeekV4TopKImpl::DeepSeekV4TopKImpl(int64_t n_routed_experts,
                                       int64_t n_activated_experts,
                                       float route_scale,
                                       int64_t vocab_size,
                                       bool use_hash,
                                       const torch::TensorOptions& options)
    : n_routed_experts_(n_routed_experts),
      n_activated_experts_(n_activated_experts),
      route_scale_(route_scale),
      vocab_size_(vocab_size),
      use_hash_(use_hash) {
  if (use_hash_) {
    tid2eid_ =
        register_parameter("tid2eid",
                           torch::empty({vocab_size_, n_activated_experts_},
                                        options.dtype(torch::kInt32)),
                           /*requires_grad=*/false);
  } else {
    bias_ = register_parameter(
        "bias",
        torch::empty({n_routed_experts_}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  }
}

DeepSeekV4TopKOutput DeepSeekV4TopKImpl::forward(
    const torch::Tensor& scores,
    const std::optional<torch::Tensor>& input_ids) {
  torch::Tensor scores_fp32 = scores.to(torch::kFloat32);
  torch::Tensor processed_scores = torch::softplus(scores_fp32).sqrt();
  torch::Tensor original_scores = processed_scores;

  torch::Tensor indices;
  if (use_hash_) {
    CHECK(input_ids.has_value()) << "input_ids is required for hash routing";
    indices = tid2eid_.index({input_ids.value()});
  } else {
    torch::Tensor biased_scores = processed_scores + bias_;
    indices = std::get<1>(biased_scores.topk(n_activated_experts_, /*dim=*/-1))
                  .to(torch::kInt32);
  }

  torch::Tensor weights = original_scores.gather(1, indices.to(torch::kInt64));
  weights = weights / weights.sum(-1, /*keepdim=*/true);
  weights = weights * route_scale_;

  return {weights, indices};
}

void DeepSeekV4TopKImpl::load_state_dict(const StateDict& state_dict) {
  if (use_hash_) {
    LOAD_WEIGHT(tid2eid);
  } else {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
