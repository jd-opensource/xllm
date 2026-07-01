/* Copyright 2025-2026 The xLLM Authors.

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

#include <ATen/ATen.h>

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

// Compute y = a @ b^T (+ bias) for 2D a:[M,K], weight b:[N,K], optional bias:[N].
//
// When `output_buf` is provided, write the result directly into that buffer
// using at::addmm_out / at::mm_out. This avoids the torch-internal at::empty
// call that F::linear performs to allocate its output tensor, which is
// forbidden during MUSA CUDA-graph capture because torch_musa's allocator
// raises "operation not permitted when stream is capturing" (it does not
// honor c10::cuda::MemPoolContext set by xLLM's graph executor).
//
// The buffer must be 2D, dtype/device-compatible with `a`, and have shape
// [M, N]. Callers (Linear layers) maintain a persistent buffer sized for the
// largest decode bucket and pass a narrow()-view sized to the current M.
//
// For non-2D inputs or when `output_buf` is unset, fall back to F::linear,
// which is fine outside graph capture.
torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias,
                     std::optional<torch::Tensor> output_buf) {
  if (output_buf.has_value() && output_buf->defined() && a.dim() == 2 &&
      b.dim() == 2) {
    auto& out = *output_buf;
    auto bt = b.t();
    if (bias.has_value() && bias->defined()) {
      at::addmm_out(out, *bias, a, bt);
    } else {
      at::mm_out(out, a, bt);
    }
    return out;
  }
  namespace F = torch::nn::functional;
  return F::linear(a, b, bias.value_or(torch::Tensor()));
}

}  // namespace xllm::kernel::cuda