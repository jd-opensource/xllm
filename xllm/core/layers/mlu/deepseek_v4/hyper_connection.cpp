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

#include "layers/mlu/deepseek_v4/hyper_connection.h"

namespace xllm {
namespace layer {

HyperConnectionHeadImpl::HyperConnectionHeadImpl(
    int64_t hc_mult,
    int64_t dim,
    float hc_eps,
    float norm_eps,
    const torch::TensorOptions& options)
    : hc_mult_(hc_mult),
      dim_(dim),
      hc_dim_(hc_mult * dim),
      hc_eps_(hc_eps),
      norm_eps_(norm_eps) {
  hc_head_fn_ = register_parameter(
      "hc_head_fn",
      torch::empty({hc_mult_, hc_dim_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hc_head_base_ = register_parameter(
      "hc_head_base",
      torch::empty({hc_mult_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hc_head_scale_ =
      register_parameter("hc_head_scale",
                         torch::empty({1}, options.dtype(torch::kFloat32)),
                         /*requires_grad=*/false);
}

torch::Tensor HyperConnectionHeadImpl::forward(const torch::Tensor& x) {
  torch::Tensor x_flat = x.flatten(-2).to(torch::kFloat32);
  torch::Tensor rsqrt =
      torch::rsqrt(x_flat.square().mean(-1, true) + norm_eps_);
  torch::Tensor mixes = torch::linear(x_flat, hc_head_fn_) * rsqrt;
  torch::Tensor pre =
      torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
  return torch::sum(pre.unsqueeze(-1) * x_flat.view(x.sizes()), -2)
      .to(x.scalar_type());
}

void HyperConnectionHeadImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(hc_head_fn);
  LOAD_WEIGHT(hc_head_base);
  LOAD_WEIGHT(hc_head_scale);
}

HyperConnectionPreImpl::HyperConnectionPreImpl(
    int64_t hc_mult,
    int64_t dim,
    int64_t hc_sinkhorn_iters,
    float hc_eps,
    float norm_eps,
    const torch::TensorOptions& options)
    : hc_mult_(hc_mult),
      dim_(dim),
      hc_dim_(hc_mult * dim),
      hc_sinkhorn_iters_(hc_sinkhorn_iters),
      hc_eps_(hc_eps),
      norm_eps_(norm_eps) {
  int64_t mix_hc = (2 + hc_mult) * hc_mult;
  hc_fn_ = register_parameter(
      "hc_fn",
      torch::empty({mix_hc, hc_dim_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hc_base_ =
      register_parameter("hc_base",
                         torch::empty({mix_hc}, options.dtype(torch::kFloat32)),
                         /*requires_grad=*/false);
  hc_scale_ =
      register_parameter("hc_scale",
                         torch::empty({3}, options.dtype(torch::kFloat32)),
                         /*requires_grad=*/false);
}

PrePostCombOutput HyperConnectionPreImpl::forward(const torch::Tensor& x) {
  torch::Tensor x_flat = x.flatten(-2).to(torch::kFloat32);
  torch::Tensor rsqrt =
      torch::rsqrt(x_flat.square().mean(-1, true) + norm_eps_);
  torch::Tensor mixes = torch::linear(x_flat, hc_fn_) * rsqrt;
  PrePostCombOutput output = hc_split_sinkhorn(mixes);
  output.pre = torch::sum(output.pre.unsqueeze(-1) * x_flat.view(x.sizes()), -2)
                   .to(x.scalar_type());
  return output;
}

PrePostCombOutput HyperConnectionPreImpl::hc_split_sinkhorn(
    const torch::Tensor& mixes) {
  int64_t hc = hc_mult_;
  int64_t total_dim = mixes.size(-1);
  CHECK_EQ(total_dim, (2 + hc) * hc);

  int64_t batch_size = mixes.size(0);
  int64_t offset = 0;

  torch::Tensor pre_logits = mixes.narrow(1, offset, hc);
  offset += hc;
  torch::Tensor post_logits = mixes.narrow(1, offset, hc);
  offset += hc;
  torch::Tensor comb_logits =
      mixes.narrow(1, offset, hc * hc).view({batch_size, hc, hc});

  torch::Tensor pre =
      torch::sigmoid(pre_logits * hc_scale_[0] + hc_base_.narrow(0, 0, hc)) +
      hc_eps_;
  torch::Tensor post = 2.0 * torch::sigmoid(post_logits * hc_scale_[1] +
                                            hc_base_.narrow(0, hc, hc));
  torch::Tensor comb_base = hc_base_.narrow(0, 2 * hc, hc * hc).view({hc, hc});
  torch::Tensor comb = comb_logits * hc_scale_[2] + comb_base;

  torch::Tensor row_max = std::get<0>(comb.max(-1, true));
  torch::Tensor comb_exp = torch::exp(comb - row_max);
  torch::Tensor row_sum = comb_exp.sum(-1, true);
  comb = comb_exp / row_sum + hc_eps_;
  torch::Tensor col_sum = comb.sum(-2, true);
  comb = comb / (col_sum + hc_eps_);

  for (int64_t i = 0; i < hc_sinkhorn_iters_ - 1; ++i) {
    comb = comb / (comb.sum(-1, true) + hc_eps_);
    comb = comb / (comb.sum(-2, true) + hc_eps_);
  }

  return {pre, post, comb};
}

void HyperConnectionPreImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(hc_fn);
  LOAD_WEIGHT(hc_base);
  LOAD_WEIGHT(hc_scale);
}

HyperConnectionPostImpl::HyperConnectionPostImpl(
    const torch::TensorOptions& options) {}

torch::Tensor HyperConnectionPostImpl::forward(const torch::Tensor& x,
                                               const torch::Tensor& residual,
                                               const torch::Tensor& post,
                                               const torch::Tensor& comb) {
  return (post.unsqueeze(-1) * x.unsqueeze(-2) +
          torch::sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), -3))
      .to(x.scalar_type());
}

}  // namespace layer
}  // namespace xllm
