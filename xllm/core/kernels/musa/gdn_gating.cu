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

// Fused Gated-Delta-Net gating: collapses the ~10-op torch sequence
//   pre  = a + dt_bias
//   g    = -exp(A_log) * softplus(pre; beta, threshold)
//   beta = sigmoid(b)
// into a single elementwise MUSA kernel. a/b: [N, H]; A_log/dt_bias: [H].

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

namespace {

template <typename scalar_t>
__global__ void gdn_gating_kernel(const scalar_t* __restrict__ a,
                                  const scalar_t* __restrict__ b,
                                  const float* __restrict__ A_log,
                                  const float* __restrict__ dt_bias,
                                  scalar_t* __restrict__ g_out,
                                  scalar_t* __restrict__ beta_out,
                                  int64_t n_elem,
                                  int H,
                                  float sp_beta,
                                  float threshold) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (idx >= n_elem) {
    return;
  }
  const int h = static_cast<int>(idx % H);

  const float pre = static_cast<float>(a[idx]) + dt_bias[h];
  // softplus(pre; beta, threshold): linear above threshold for stability.
  const float bx = sp_beta * pre;
  const float sp = (bx > threshold) ? pre : (log1pf(expf(bx)) / sp_beta);
  const float g = -expf(A_log[h]) * sp;
  const float beta = 1.f / (1.f + expf(-static_cast<float>(b[idx])));

  g_out[idx] = static_cast<scalar_t>(g);
  beta_out[idx] = static_cast<scalar_t>(beta);
}

template <typename scalar_t>
void launch(const torch::Tensor& a,
            const torch::Tensor& b,
            const torch::Tensor& A_log_f32,
            const torch::Tensor& dt_bias_f32,
            torch::Tensor& g,
            torch::Tensor& beta,
            int64_t n_elem,
            int H,
            float sp_beta,
            float threshold,
            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n_elem + threads - 1) / threads);
  gdn_gating_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
      A_log_f32.data_ptr<float>(), dt_bias_f32.data_ptr<float>(),
      g.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), n_elem, H, sp_beta,
      threshold);
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> gdn_gating(const torch::Tensor& a,
                                                   const torch::Tensor& b,
                                                   const torch::Tensor& A_log,
                                                   const torch::Tensor& dt_bias,
                                                   double sp_beta,
                                                   double threshold) {
  const int64_t H = a.size(-1);
  const int64_t n_elem = a.numel();
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto A_log_f32 = A_log.to(torch::kFloat32).contiguous();
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32).contiguous();
  auto g = torch::empty_like(a_c);
  auto beta = torch::empty_like(a_c);

  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (a_c.scalar_type() == torch::kFloat32) {
    launch<float>(a_c, b_c, A_log_f32, dt_bias_f32, g, beta, n_elem,
                  static_cast<int>(H), static_cast<float>(sp_beta),
                  static_cast<float>(threshold), stream);
  } else if (a_c.scalar_type() == torch::kBFloat16) {
    launch<at::BFloat16>(a_c, b_c, A_log_f32, dt_bias_f32, g, beta, n_elem,
                         static_cast<int>(H), static_cast<float>(sp_beta),
                         static_cast<float>(threshold), stream);
  } else {
    TORCH_CHECK(false, "gdn_gating: unsupported dtype ", a_c.scalar_type());
  }
  return {g, beta};
}

}  // namespace xllm::kernel::cuda
