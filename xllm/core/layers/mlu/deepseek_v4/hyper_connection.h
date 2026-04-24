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

#pragma once

#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class HyperConnectionHeadImpl : public torch::nn::Module {
 public:
  HyperConnectionHeadImpl() = default;
  HyperConnectionHeadImpl(int64_t hc_mult,
                          int64_t dim,
                          float hc_eps,
                          float norm_eps,
                          const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& x);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t hc_mult_;
  int64_t dim_;
  int64_t hc_dim_;
  float hc_eps_;
  float norm_eps_;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(HyperConnectionHead);

struct PrePostCombOutput {
  torch::Tensor pre;
  torch::Tensor post;
  torch::Tensor comb;
};

class HyperConnectionPreImpl : public torch::nn::Module {
 public:
  HyperConnectionPreImpl() = default;
  HyperConnectionPreImpl(int64_t hc_mult,
                         int64_t dim,
                         int64_t hc_sinkhorn_iters,
                         float hc_eps,
                         float norm_eps,
                         const torch::TensorOptions& options);

  PrePostCombOutput forward(const torch::Tensor& x);

  void load_state_dict(const StateDict& state_dict);

 private:
  PrePostCombOutput hc_split_sinkhorn(const torch::Tensor& mixes);

  int64_t hc_mult_;
  int64_t dim_;
  int64_t hc_dim_;
  int64_t hc_sinkhorn_iters_;
  float hc_eps_;
  float norm_eps_;

  DEFINE_WEIGHT(hc_fn);
  DEFINE_WEIGHT(hc_base);
  DEFINE_WEIGHT(hc_scale);
};
TORCH_MODULE(HyperConnectionPre);

class HyperConnectionPostImpl : public torch::nn::Module {
 public:
  HyperConnectionPostImpl() = default;
  explicit HyperConnectionPostImpl(const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& residual,
                        const torch::Tensor& post,
                        const torch::Tensor& comb);
};
TORCH_MODULE(HyperConnectionPost);

}  // namespace layer
}  // namespace xllm
