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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.h"

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

class KVCacheShape;

struct KVCacheCapacity {
  PROPERTY(int64_t, n_blocks) = 0;
  PROPERTY(int64_t, cache_size_in_bytes) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, slot_size) = 0;

  // for index cache
  PROPERTY(int64_t, index_slot_size) = 0;

  // for kv cache quantization scale cache
  PROPERTY(int64_t, scale_slot_size) = 0;

  // for linear attention
  PROPERTY(int64_t, linear_slot_size) = 0;
  PROPERTY(int64_t, linear_cache_size_in_bytes) = 0;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, num_linear_state_blocks) = 0;
  PROPERTY(int64_t, num_full_attention_layers) = 0;
  PROPERTY(int64_t, num_linear_attention_layers) = 0;

  // DeepSeek V4 uses separate block pools for sliding-window and compressed
  // caches. These fields are only meaningful for deepseek_v4.
  PROPERTY(int64_t, swa_count) = 0;
  PROPERTY(int64_t, c4_count) = 0;
  PROPERTY(int64_t, c128_count) = 0;
};

struct KVCacheCreateOptions {
  PROPERTY(torch::Device, device) = torch::Device(torch::kCPU);
  // kvcache dtype for key/value cacahe, index cache
  PROPERTY(torch::ScalarType, dtype) = torch::kBFloat16;
  // ssm dtype for linear attention layers
  PROPERTY(torch::ScalarType, ssm_dtype) = torch::kBFloat16;
  PROPERTY(int64_t, num_layers) = 0;
  // full attention interval for linear attention layers
  PROPERTY(int64_t, full_attention_interval) = 1;
  // model_id are required for XTensor mode
  PROPERTY(std::string, model_id);
  PROPERTY(std::string, model_type);
  PROPERTY(bool, enable_xtensor) = false;
  PROPERTY(bool, enable_linear_attention) = false;
  PROPERTY(bool, enable_lighting_indexer) = false;
  PROPERTY(bool, enable_kv_cache_quant) = false;
#if defined(USE_NPU)
  PROPERTY(bool, enable_kv_cache_huge_page_allocator) = false;
#endif

  // DeepSeek V4 cache allocation metadata.
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, index_head_dim) = 0;
  PROPERTY(int64_t, window_size) = 0;
  PROPERTY(std::vector<int32_t>, compress_ratios);
};

struct KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor value_cache;
};

struct IndexedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor index_cache;
};

struct QuantizedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor key_cache_scale;
  torch::Tensor value_cache_scale;
};

struct LinearAttentionKVCacheTensors {
  torch::Tensor conv_cache;
  torch::Tensor ssm_cache;
};

struct KVCacheTensor {
  KVCacheTensorRole role;
  torch::Tensor tensor;
};

struct DeepSeekV4KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor index_cache;
  torch::Tensor indexer_cache_scale;
  torch::Tensor swa_cache;
  torch::Tensor compress_kv_state;
  torch::Tensor compress_score_state;
  torch::Tensor compress_index_kv_state;
  torch::Tensor compress_index_score_state;
};

struct DeepSeekV4CachePolicy {
  torch::ScalarType index_dtype = torch::kInt8;
  int64_t index_dtype_size = 1;
  bool has_indexer_cache_scale = true;
  torch::ScalarType scale_dtype = torch::kFloat16;
  int64_t scale_dtype_size = 2;
};

inline int64_t get_dsv4_dtype_size(torch::ScalarType dtype) {
  return static_cast<int64_t>(torch::scalarTypeToTypeMeta(dtype).itemsize());
}

inline DeepSeekV4CachePolicy get_dsv4_cache_policy(
    torch::ScalarType model_dtype) {
  DeepSeekV4CachePolicy policy;
#if defined(USE_MLU)
  policy.index_dtype = model_dtype;
  policy.index_dtype_size = get_dsv4_dtype_size(model_dtype);
  policy.has_indexer_cache_scale = false;
  policy.scale_dtype_size = 0;
#else
  policy.index_dtype_size = get_dsv4_dtype_size(policy.index_dtype);
  policy.scale_dtype_size = get_dsv4_dtype_size(policy.scale_dtype);
#endif
  return policy;
}

// for qwen3.5
bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval);

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type);
#endif

}  // namespace xllm
