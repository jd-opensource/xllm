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

namespace xllm {

class KVCacheShape;

struct KVCacheCapacity {
  PROPERTY(int64_t, n_blocks) = 0;
  PROPERTY(int64_t, cache_size_in_bytes) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, slot_size) = 0;

  // for index cache
  PROPERTY(int64_t, index_slot_size) = 0;

  // for linear attention
  PROPERTY(int64_t, linear_slot_size) = 0;
  PROPERTY(int64_t, linear_cache_size_in_bytes) = 0;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, num_linear_state_blocks) = 0;
  PROPERTY(int64_t, num_full_attention_layers) = 0;
  PROPERTY(int64_t, num_linear_attention_layers) = 0;
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
