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

#include "framework/kv_cache/kv_cache.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/kv_cache/indexed_kv_cache_impl.h"
#include "framework/kv_cache/linear_attention_kv_cache_impl.h"
#include "framework/kv_cache/quantized_kv_cache_impl.h"
#include "framework/xtensor/xtensor_allocator.h"

namespace xllm {

namespace {

std::unique_ptr<KVCacheImpl> create_kv_cache_impl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_id) {
  CHECK_GE(layer_id, 0) << "KV cache layer_id must be non-negative.";

#if !defined(USE_MLU)
  CHECK(!create_options.enable_kv_cache_quant())
      << "KV cache quantization is only supported on MLU backend.";
#endif

  const bool is_linear_layer =
      create_options.enable_linear_attention() &&
      is_linear_attention_layer(layer_id,
                                create_options.full_attention_interval());
  if (is_linear_layer) {
    return std::make_unique<LinearAttentionKVCacheImpl>(kv_cache_shape,
                                                        create_options);
  }

  if (create_options.enable_kv_cache_quant() &&
      !create_options.enable_lighting_indexer()) {
    return std::make_unique<QuantizedKVCacheImpl>(kv_cache_shape,
                                                  create_options);
  }

  if (create_options.enable_lighting_indexer()) {
    return std::make_unique<IndexedKVCacheImpl>(kv_cache_shape, create_options);
  }

  return std::make_unique<KVCacheImpl>(kv_cache_shape, create_options);
}

}  // namespace

KVCache::KVCache() : impl_(std::make_unique<KVCacheImpl>()) {}

KVCache::KVCache(const KVCacheTensors& tensors)
    : impl_(std::make_unique<KVCacheImpl>(tensors)) {}

KVCache::KVCache(const IndexedKVCacheTensors& tensors)
    : impl_(std::make_unique<IndexedKVCacheImpl>(tensors)) {}

KVCache::KVCache(const LinearAttentionKVCacheTensors& tensors)
    : impl_(std::make_unique<LinearAttentionKVCacheImpl>(tensors)) {}

KVCache::KVCache(const QuantizedKVCacheTensors& tensors)
    : impl_(std::make_unique<QuantizedKVCacheImpl>(tensors)) {}

KVCache::KVCache(const KVCacheShape& kv_cache_shape,
                 const KVCacheCreateOptions& create_options,
                 int64_t layer_id)
    : impl_(create_kv_cache_impl(kv_cache_shape, create_options, layer_id)) {}

torch::Tensor KVCache::get_k_cache() const { return impl_->get_k_cache(); }
torch::Tensor KVCache::get_v_cache() const { return impl_->get_v_cache(); }
torch::Tensor KVCache::get_index_cache() const {
  return impl_->get_index_cache();
}
torch::Tensor KVCache::get_conv_cache() const {
  return impl_->get_conv_cache();
}
torch::Tensor KVCache::get_ssm_cache() const { return impl_->get_ssm_cache(); }

std::optional<torch::Tensor> KVCache::get_k_cache_scale() const {
  return impl_->get_k_cache_scale();
}

std::optional<torch::Tensor> KVCache::get_v_cache_scale() const {
  return impl_->get_v_cache_scale();
}

std::vector<std::vector<int64_t>> KVCache::get_shapes() {
  return impl_->get_shapes();
}

bool KVCache::empty() const { return impl_->empty(); }

void KVCache::swap_blocks(torch::Tensor& src_tensor,
                          torch::Tensor& dst_tensor) {
  impl_->swap_blocks(src_tensor, dst_tensor);
}

void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options) {
  CHECK(kv_caches.empty()) << "KV caches are already initialized.";

  const int64_t num_layers = create_options.num_layers();
  kv_caches.reserve(num_layers);

  if (create_options.enable_xtensor()) {
    CHECK(kv_cache_shape.has_key_cache_shape())
        << "key_cache_shape must be initialized for XTensor mode.";
    CHECK(kv_cache_shape.has_value_cache_shape())
        << "value_cache_shape must be initialized for XTensor mode.";
    CHECK(!kv_cache_shape.has_index_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!kv_cache_shape.has_conv_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!kv_cache_shape.has_ssm_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!create_options.model_id().empty())
        << "model_id must not be empty for XTensor mode.";
    CHECK(!create_options.enable_linear_attention())
        << "Linear attention is not supported for XTensor mode.";

    XTensorAllocator& allocator = XTensorAllocator::get_instance();
    std::vector<torch::Tensor> k_tensors =
        allocator.create_k_tensors(create_options.model_id(),
                                   kv_cache_shape.key_cache_shape(),
                                   create_options.dtype(),
                                   num_layers);
    std::vector<torch::Tensor> v_tensors =
        allocator.create_v_tensors(create_options.model_id(),
                                   kv_cache_shape.value_cache_shape(),
                                   create_options.dtype(),
                                   num_layers);

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      torch::Tensor k_tensor = k_tensors[layer_idx];
      torch::Tensor v_tensor = v_tensors[layer_idx];
#if defined(USE_NPU)
      k_tensor = at_npu::native::npu_format_cast(k_tensor, ACL_FORMAT_ND);
      v_tensor = at_npu::native::npu_format_cast(v_tensor, ACL_FORMAT_ND);
#endif
      kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
    }
    return;
  }

  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    kv_caches.emplace_back(kv_cache_shape, create_options, layer_idx);
  }
}

}  // namespace xllm
