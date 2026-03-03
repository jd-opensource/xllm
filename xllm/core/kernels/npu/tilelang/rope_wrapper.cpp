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

#include "rope_wrapper.h"

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <limits>

#include "acl/acl.h"

#ifndef XLLM_TL_ROPE_HEAD_DIM
#define XLLM_TL_ROPE_HEAD_DIM 128
#endif

#ifndef XLLM_TL_ROPE_ROPE_DIM
#define XLLM_TL_ROPE_ROPE_DIM 128
#endif

#ifndef XLLM_TL_ROPE_ENTRY
#define XLLM_TL_ROPE_ENTRY call
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

// Generated in build dir by tilelang codegen, then compiled by bisheng.
extern "C" void XLLM_TL_ROPE_ENTRY(uint8_t* x_handle,
                                   uint8_t* sin_handle,
                                   uint8_t* cos_handle,
                                   uint8_t* out_handle,
                                   int num_tokens,
                                   int x_stride,
                                   aclrtStream stream);

void check_supported(const torch::Tensor& input,
                     const torch::Tensor& sin_cache,
                     const torch::Tensor& cos_cache) {
  CHECK(input.defined()) << "TileLang RoPE: input must be defined";
  CHECK(sin_cache.defined()) << "TileLang RoPE: sin_cache must be defined";
  CHECK(cos_cache.defined()) << "TileLang RoPE: cos_cache must be defined";

  CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
        sin_cache.device().type() == c10::DeviceType::PrivateUse1 &&
        cos_cache.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang RoPE: all tensors must be on NPU";

  CHECK_EQ(input.dtype(), torch::kBFloat16)
      << "TileLang RoPE: input must be bfloat16";
  CHECK_EQ(sin_cache.dtype(), torch::kBFloat16)
      << "TileLang RoPE: sin_cache must be bfloat16";
  CHECK_EQ(cos_cache.dtype(), torch::kBFloat16)
      << "TileLang RoPE: cos_cache must be bfloat16";

  CHECK_EQ(input.dim(), 3) << "TileLang RoPE: input must be 3D [T, H, D]";
  CHECK_EQ(input.size(2), XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: input last dim must equal rope_dim("
      << XLLM_TL_ROPE_ROPE_DIM << ")";
  CHECK_EQ(input.stride(2), 1)
      << "TileLang RoPE: input last dim stride must be 1";
  CHECK_EQ(input.stride(0), input.size(1) * input.stride(1))
      << "TileLang RoPE: unsupported input layout";

  CHECK_EQ(sin_cache.dim(), 2)
      << "TileLang RoPE: sin_cache must be 2D [T, rope_dim]";
  CHECK_EQ(cos_cache.dim(), 2)
      << "TileLang RoPE: cos_cache must be 2D [T, rope_dim]";
  CHECK_EQ(sin_cache.sizes(), cos_cache.sizes())
      << "TileLang RoPE: sin_cache/cos_cache shape mismatch";
  CHECK_EQ(sin_cache.size(1), XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: rope_dim mismatch, expected " << XLLM_TL_ROPE_ROPE_DIM;
  CHECK_EQ(sin_cache.size(0), input.size(0))
      << "TileLang RoPE: sin_cache token size must match input.size(0)";

  const int64_t row_count = input.size(0) * input.size(1);
  CHECK_GT(row_count, 0) << "TileLang RoPE: row_count must be > 0";
}

void run_tilelang_rope_once(torch::Tensor& x_rows,
                            const torch::Tensor& sin_rows,
                            const torch::Tensor& cos_rows) {
  CHECK_EQ(x_rows.dim(), 2) << "TileLang RoPE: x_rows must be 2D";
  CHECK_EQ(sin_rows.dim(), 2) << "TileLang RoPE: sin_rows must be 2D";
  CHECK_EQ(cos_rows.dim(), 2) << "TileLang RoPE: cos_rows must be 2D";
  CHECK_EQ(x_rows.size(0), sin_rows.size(0))
      << "TileLang RoPE: x_rows/sin_rows row mismatch";
  CHECK_EQ(x_rows.size(0), cos_rows.size(0))
      << "TileLang RoPE: x_rows/cos_rows row mismatch";
  CHECK_EQ(x_rows.size(1), XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: x_rows rope_dim mismatch";
  CHECK_EQ(sin_rows.size(1), XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: sin_rows rope_dim mismatch";
  CHECK_EQ(cos_rows.size(1), XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: cos_rows rope_dim mismatch";

  CHECK_EQ(x_rows.stride(1), 1)
      << "TileLang RoPE: x_rows last dim stride must be 1";
  CHECK(sin_rows.is_contiguous())
      << "TileLang RoPE: sin_rows must be contiguous";
  CHECK(cos_rows.is_contiguous())
      << "TileLang RoPE: cos_rows must be contiguous";

  const int64_t row_count = x_rows.size(0);
  CHECK_LE(row_count, static_cast<int64_t>(std::numeric_limits<int>::max()))
      << "TileLang RoPE: row_count exceeds int range";
  CHECK_GE(XLLM_TL_ROPE_HEAD_DIM, XLLM_TL_ROPE_ROPE_DIM)
      << "TileLang RoPE: compiled head_dim must be >= rope_dim";

  const int32_t device_id = x_rows.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  const int num_tokens = static_cast<int>(row_count);
  const int64_t x_stride64 = x_rows.stride(0);
  CHECK_GT(x_stride64, 0) << "TileLang RoPE: x_rows stride must be > 0";
  CHECK_LE(x_stride64, static_cast<int64_t>(XLLM_TL_ROPE_HEAD_DIM))
      << "TileLang RoPE: x_rows stride exceeds compiled head_dim("
      << XLLM_TL_ROPE_HEAD_DIM << ")";
  CHECK_LE(x_stride64, static_cast<int64_t>(std::numeric_limits<int>::max()))
      << "TileLang RoPE: x_rows stride exceeds int range";
  const int x_stride = static_cast<int>(x_stride64);

  // Kernel is RoPE-in-place: pass x_rows as both input and output.
  XLLM_TL_ROPE_ENTRY(
      reinterpret_cast<uint8_t*>(x_rows.data_ptr()),
      reinterpret_cast<uint8_t*>(const_cast<void*>(sin_rows.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(cos_rows.data_ptr())),
      reinterpret_cast<uint8_t*>(x_rows.data_ptr()),
      num_tokens,
      x_stride,
      stream);
}

}  // namespace

void rope_in_place(torch::Tensor& input,
                   const torch::Tensor& sin_cache,
                   const torch::Tensor& cos_cache) {
  check_supported(input, sin_cache, cos_cache);

  auto input_rows =
      input.as_strided({input.size(0) * input.size(1), input.size(2)},
                       {input.stride(1), input.stride(2)});
  auto sin_rows = sin_cache.unsqueeze(1)
                      .expand({input.size(0), input.size(1), sin_cache.size(1)})
                      .contiguous()
                      .view({input_rows.size(0), sin_cache.size(1)});
  auto cos_rows = cos_cache.unsqueeze(1)
                      .expand({input.size(0), input.size(1), cos_cache.size(1)})
                      .contiguous()
                      .view({input_rows.size(0), cos_cache.size(1)});
  run_tilelang_rope_once(input_rows, sin_rows, cos_rows);
}

}  // namespace xllm::kernel::npu::tilelang
