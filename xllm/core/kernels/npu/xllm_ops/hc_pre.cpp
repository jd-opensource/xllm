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

#include <torch/library.h>

#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x.dim(),
              ".");

  auto original_type = x.dtype();
  auto x_bf16 = x;
  if (x_bf16.dtype() != at::kBFloat16) {
    x_bf16 = x_bf16.to(at::kBFloat16);
  }

  auto rsqrt = hc_pre_inv_rms(x_bf16, norm_eps);
  at::Tensor x_float = x_bf16.to(at::kFloat);
  at::Tensor x_flattened =
      x.dim() == 4 ? x_float.flatten(2, -1) : x_float.flatten(1, -1);
  auto mixes = at::linear(x_flattened, hc_fn);

  auto output = hc_pre_sinkhorn(mixes,
                                rsqrt,
                                hc_scale,
                                hc_base,
                                x_bf16,
                                hc_mult,
                                hc_sinkhorn_iters,
                                hc_eps);

  at::Tensor y = std::get<0>(output).to(original_type);
  return std::make_tuple(y, std::get<1>(output), std::get<2>(output));
}

}  // namespace xllm::kernel::npu
