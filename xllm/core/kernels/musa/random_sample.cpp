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

#include <cstdint>

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

// Categorical sampling from a probability distribution.
//
// NOTE (MUSA bring-up): the previous implementation called the mate
// `sampling_from_probs` SO and obtained a philox seed/offset from the CUDA
// default generator. Both are unavailable on MUSA-as-CUDA:
//   * at::globalContext().lazyInitCUDA() / getDefaultCUDAGenerator() abort with
//     "Cannot initialize CUDA without ATen_cuda library";
//   * mate_cached_ops/sampling/sampling.so is not deployed.
// We instead perform inverse-CDF sampling with plain torch ops and draw the
// uniforms on the HOST (CPU generator), which works on every backend.
torch::Tensor random_sample(const torch::Tensor& probs) {
  CHECK(probs.dim() == 2 || probs.dim() == 3)
      << "probs must be a 2D or 3D tensor";

  torch::Tensor flat_probs =
      (probs.dim() == 3) ? probs.reshape({-1, probs.size(2)}) : probs;

  const torch::Device device = flat_probs.device();
  // Input is already the softmax distribution (float32, normalized to sum=1 by
  // the sampler), so we skip the redundant cast/renormalization passes.
  auto p = (flat_probs.scalar_type() == torch::kFloat32)
               ? flat_probs
               : flat_probs.to(torch::kFloat32);

  const int64_t batch_size = p.size(0);
  const int64_t vocab_size = p.size(1);

  auto cdf = p.cumsum(/*dim=*/-1);

  // Uniforms drawn on the host to avoid device RNG init (lazyInitCUDA aborts on
  // MUSA). Scale by the total mass (last cdf column) instead of a separate
  // normalization pass; also stays robust if probs don't sum to exactly 1.
  auto u = torch::rand({batch_size, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
               .to(device);
  u = u * cdf.narrow(/*dim=*/-1, vocab_size - 1, 1);

  // Inverse-CDF sample = first index i with cdf[i] >= u, via binary search
  // (O(log vocab)) instead of an O(vocab) compare-and-sum.
  auto samples = torch::searchsorted(cdf, u).squeeze(-1).to(torch::kInt64);
  samples = samples.clamp(/*min=*/0, /*max=*/vocab_size - 1);

  if (probs.dim() == 3) {
    return samples.reshape({probs.size(0), probs.size(1)});
  }
  return samples.flatten();
}

}  // namespace xllm::kernel::cuda
