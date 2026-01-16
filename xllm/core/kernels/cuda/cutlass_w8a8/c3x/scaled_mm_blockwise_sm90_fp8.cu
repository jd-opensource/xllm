/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8.cu

// clang-format off
#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm90_fp8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
// clang-format on

namespace xllm {
namespace kernel {
namespace cuda {

void cutlass_scaled_mm_blockwise_sm90_fp8(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          torch::Tensor const& a_scales,
                                          torch::Tensor const& b_scales) {
  if (out.dtype() == torch::kBFloat16) {
    cutlass_gemm_blockwise_sm90_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);

  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    cutlass_gemm_blockwise_sm90_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm