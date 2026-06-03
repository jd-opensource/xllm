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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>
#include <string>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_CAUSAL_CONV1D_PREFILL_REGISTRY_INC
#error "XLLM_TL_CAUSAL_CONV1D_PREFILL_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_CAUSAL_CONV1D_PREFILL_REGISTRY_INC

CausalConv1dPrefillSpecialization build_prefill_runtime_specialization(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& cu_seqlens,
    int64_t original_dim,
    bool has_silu) {
  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d_prefill: x must be 2D [T, D]";
  CHECK_GE(weight.dim(), 2)
      << "TileLang causal_conv1d_prefill: weight must be >=2D [width, dim]";
  CHECK_EQ(cu_seqlens.dim(), 1)
      << "TileLang causal_conv1d_prefill: cu_seqlens must be 1D";
  CHECK_GE(cu_seqlens.size(0), 2) << "TileLang causal_conv1d_prefill: "
                                     "cu_seqlens must have at least 2 elements";

  const int32_t num_batches = static_cast<int32_t>(cu_seqlens.size(0) - 1);
  const int32_t dim = static_cast<int32_t>(original_dim);
  const int32_t width = static_cast<int32_t>(weight.size(0));

  CHECK_GT(num_batches, 0)
      << "TileLang causal_conv1d_prefill: num_batches must be > 0";
  CHECK_GT(dim, 0) << "TileLang causal_conv1d_prefill: dim must be > 0";
  CHECK_GT(width, 0) << "TileLang causal_conv1d_prefill: width must be > 0";

  return make_causal_conv1d_prefill_specialization(
      CausalConv1dPrefillNumBatches{num_batches},
      CausalConv1dPrefillDim{dim},
      CausalConv1dPrefillWidth{width},
      CausalConv1dPrefillHasSilu{static_cast<int32_t>(has_silu)},
      CausalConv1dPrefillDType{to_tilelang_dtype(
          x.scalar_type() == c10::ScalarType::BFloat16 ? c10::ScalarType::Half
                                                       : x.scalar_type())});
}

}  // namespace

bool has_causal_conv1d_prefill_specialization(int64_t num_batches,
                                              int64_t dim,
                                              bool has_silu) {
  const int32_t width = 4;
  CausalConv1dPrefillSpecialization spec =
      make_causal_conv1d_prefill_specialization(
          CausalConv1dPrefillNumBatches{static_cast<int32_t>(num_batches)},
          CausalConv1dPrefillDim{static_cast<int32_t>(dim)},
          CausalConv1dPrefillWidth{width},
          CausalConv1dPrefillHasSilu{static_cast<int32_t>(has_silu)},
          CausalConv1dPrefillDType{TilelangDType::kFloat16});
  return find_causal_conv1d_prefill_kernel_entry(spec) != nullptr;
}

torch::Tensor causal_conv1d_prefill(torch::Tensor& conv_state,
                                    const torch::Tensor& x,
                                    const torch::Tensor& weight,
                                    const torch::Tensor& bias,
                                    const torch::Tensor& cu_seqlens,
                                    const torch::Tensor& init_indices,
                                    const torch::Tensor& current_indices,
                                    const torch::Tensor& initial_state_mode,
                                    bool has_silu) {
  CHECK(x.defined()) << "TileLang causal_conv1d_prefill: x must be defined";
  CHECK(conv_state.defined())
      << "TileLang causal_conv1d_prefill: conv_state must be defined";
  CHECK(weight.defined())
      << "TileLang causal_conv1d_prefill: weight must be defined";
  CHECK(bias.defined())
      << "TileLang causal_conv1d_prefill: bias must be defined";
  CHECK(cu_seqlens.defined())
      << "TileLang causal_conv1d_prefill: cu_seqlens must be defined";
  CHECK(init_indices.defined())
      << "TileLang causal_conv1d_prefill: init_indices must be defined";
  CHECK(current_indices.defined())
      << "TileLang causal_conv1d_prefill: current_indices must be defined";
  CHECK(initial_state_mode.defined())
      << "TileLang causal_conv1d_prefill: initial_state_mode must be defined";

  CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
        conv_state.device().type() == c10::DeviceType::PrivateUse1 &&
        weight.device().type() == c10::DeviceType::PrivateUse1 &&
        bias.device().type() == c10::DeviceType::PrivateUse1 &&
        cu_seqlens.device().type() == c10::DeviceType::PrivateUse1 &&
        init_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        current_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        initial_state_mode.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang causal_conv1d_prefill: all tensors must be on NPU";

  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d_prefill: x must be 2D [T, D]";
  CHECK_EQ(conv_state.dim(), 3)
      << "TileLang causal_conv1d_prefill: conv_state must be 3D [C, SL, D]";
  CHECK_GE(weight.dim(), 2)
      << "TileLang causal_conv1d_prefill: weight must be >=2D";
  CHECK_EQ(bias.dim(), 1)
      << "TileLang causal_conv1d_prefill: bias must be 1D [D]";
  CHECK_EQ(cu_seqlens.dim(), 1)
      << "TileLang causal_conv1d_prefill: cu_seqlens must be 1D";
  CHECK_EQ(init_indices.dim(), 1)
      << "TileLang causal_conv1d_prefill: init_indices must be 1D";
  CHECK_EQ(current_indices.dim(), 1)
      << "TileLang causal_conv1d_prefill: current_indices must be 1D";
  CHECK_EQ(initial_state_mode.dim(), 1)
      << "TileLang causal_conv1d_prefill: initial_state_mode must be 1D";

  const int64_t dim = x.size(1);
  const int64_t num_batches = cu_seqlens.size(0) - 1;

  CHECK_EQ(conv_state.size(2), dim)
      << "TileLang causal_conv1d_prefill: conv_state dim mismatch";
  CHECK_EQ(bias.size(0), dim)
      << "TileLang causal_conv1d_prefill: bias dim mismatch";
  CHECK_EQ(init_indices.size(0), num_batches)
      << "TileLang causal_conv1d_prefill: init_indices batch mismatch";
  CHECK_EQ(current_indices.size(0), num_batches)
      << "TileLang causal_conv1d_prefill: current_indices batch mismatch";
  CHECK_EQ(initial_state_mode.size(0), num_batches)
      << "TileLang causal_conv1d_prefill: initial_state_mode batch mismatch";

  CHECK_EQ(x.dtype(), conv_state.dtype())
      << "TileLang causal_conv1d_prefill: x/conv_state dtype mismatch";
  CHECK_EQ(x.dtype(), weight.dtype())
      << "TileLang causal_conv1d_prefill: x/weight dtype mismatch";
  CHECK_EQ(x.dtype(), bias.dtype())
      << "TileLang causal_conv1d_prefill: x/bias dtype mismatch";

  CHECK_EQ(cu_seqlens.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_prefill: cu_seqlens must be int32";
  CHECK_EQ(init_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_prefill: init_indices must be int32";
  CHECK_EQ(current_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_prefill: current_indices must be int32";
  CHECK_EQ(initial_state_mode.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_prefill: initial_state_mode must be int32";

  CHECK(x.is_contiguous())
      << "TileLang causal_conv1d_prefill: x must be contiguous";
  CHECK(conv_state.is_contiguous())
      << "TileLang causal_conv1d_prefill: conv_state must be contiguous";
  CHECK(weight.is_contiguous())
      << "TileLang causal_conv1d_prefill: weight must be contiguous";
  CHECK(bias.is_contiguous())
      << "TileLang causal_conv1d_prefill: bias must be contiguous";
  CHECK(cu_seqlens.is_contiguous())
      << "TileLang causal_conv1d_prefill: cu_seqlens must be contiguous";
  CHECK(init_indices.is_contiguous())
      << "TileLang causal_conv1d_prefill: init_indices must be contiguous";
  CHECK(current_indices.is_contiguous())
      << "TileLang causal_conv1d_prefill: current_indices must be contiguous";
  CHECK(initial_state_mode.is_contiguous())
      << "TileLang causal_conv1d_prefill: initial_state_mode must be "
         "contiguous";

  const auto specialization = build_prefill_runtime_specialization(
      x, weight, cu_seqlens, dim, has_silu);
  const auto* entry = find_causal_conv1d_prefill_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang causal_conv1d_prefill: no compiled variant. "
      << "Available variants: "
      << available_causal_conv1d_prefill_variant_keys();

  auto y = torch::empty_like(x);

  const int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(weight.data_ptr())),
      reinterpret_cast<uint8_t*>(conv_state.data_ptr()),
      reinterpret_cast<uint8_t*>(const_cast<void*>(cu_seqlens.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(init_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(current_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(initial_state_mode.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(bias.data_ptr())),
      reinterpret_cast<uint8_t*>(y.data_ptr()),
      static_cast<int64_t>(x.size(0)),
      static_cast<int64_t>(x.size(1)),
      static_cast<int64_t>(conv_state.size(0)),
      static_cast<int64_t>(conv_state.size(1)),
      stream);

  return y;
}

}  // namespace xllm::kernel::npu::tilelang
