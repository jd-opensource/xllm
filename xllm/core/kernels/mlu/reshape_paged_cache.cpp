/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "torch_mlu/csrc/aten/cnnl/cnnlHandle.h"
#include "torch_mlu/csrc/framework/core/mlu_guard.h"
#include "torch_mlu_ops.h"
#include "torch_ops_api.h"
#include "utils.h"

namespace xllm::mlu {

void reshape_paged_cache(
    torch::Tensor& k,                       // [num_tokens, num_heads, head_dim]
    const c10::optional<torch::Tensor>& v,  // [num_tokens, num_heads, head_dim]
    torch::Tensor& k_cache,  // [num_blocks, num_heads, block_size, head_dim]
    const c10::optional<torch::Tensor>&
        v_cache,  // [num_blocks, num_heads, block_size, head_dim]
    const torch::Tensor& slot_mapping,
    bool direction) {
  // 1. check device and tensor type
  checkTensorSameAttr<TensorAttr::ALL>(k, v, k_cache, v_cache);
  checkTensorSameAttr<TensorAttr::DEVICE>(k, slot_mapping);
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt32,
              "slot_mapping type need be int32");

  // 2. check shape
  const int num_tokens = k.size(0);
  const int num_heads = k.size(1);
  const int head_dim = k.size(2);
  const int block_size = k_cache.size(2);
  const int head_size = k_cache.size(3);
  const int num_blocks = k_cache.size(0);

  CHECK_SHAPE(k, num_tokens, num_heads, head_dim);
  CHECK_SHAPE(k_cache, num_blocks, num_heads, block_size, head_dim);
  TORCH_CHECK(v.has_value() == v_cache.has_value(),
              "v.has_value() == v_cache.has_value().")
  if (v.has_value()) {
    CHECK_SHAPE(v.value(), num_tokens, num_heads, head_dim);
    CHECK_SHAPE(v_cache.value(), num_blocks, num_heads, block_size, head_dim);
  }
  CHECK_SHAPE(slot_mapping, num_tokens);

  // 3. check strides
  checkTensorContiguous("k_cache, v_cache, slot_mapping must be contiguous.",
                        k_cache,
                        v_cache,
                        slot_mapping);
  TORCH_CHECK(k.stride(-1) == 1, "k last dim must be contiguous.");
  TORCH_CHECK(k.stride(-2) == head_dim,
              "k last second dim must be contiguous.");
  if (v.has_value()) {
    TORCH_CHECK(v.value().stride(-1) == 1, "v last dim must be contiguous.");
    TORCH_CHECK(v.value().stride(-2) == head_dim,
                "v last second dim must be contiguous.");
  }

  const torch_mlu::mlu::MLUGuard device_guard(k.device());
  auto queue = torch_mlu::getCurMLUStream();
  cnnlDataType_t dtype = getCnnlDataType(k.scalar_type());
  size_t key_stride0 = static_cast<size_t>(k.stride(0));
  size_t value_stride0 = 1;
  if (v.has_value()) {
    value_stride0 = static_cast<size_t>(v.value().stride(0));
  }

  TMO_KERNEL_CHECK_FATAL(tmo::invokeReshapePagedCache(
      queue,
      dtype,
      getAtTensorPtr(k),
      getAtTensorPtr(v),
      getAtTensorPtr(k_cache),
      getAtTensorPtr(v_cache),
      nullptr,
      getAtTensorPtr(slot_mapping),
      0,
      0,
      key_stride0,
      value_stride0,
      num_tokens,
      num_heads,
      num_blocks,
      block_size,
      head_size,
      direction ? tmo::CACHE2CONTEXT : tmo::CACHE2CONTEXT));
}

}  // namespace xllm::mlu
