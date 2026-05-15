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

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

torch::Tensor swap_tensor_blocks(const torch::Tensor& tensor,
                                 const torch::Tensor& src_tensor,
                                 const torch::Tensor& dst_tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  auto selected = torch::index_select(tensor, 0, src_tensor);
  auto result = tensor.clone();
  result.index_copy_(0, dst_tensor, selected);
  return result;
}

std::string tensor_shape_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}

std::string int32_vector_string(const std::vector<int32_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

torch::Tensor cast_to_nd_format(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
#if defined(USE_NPU)
  return at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND);
#else
  return tensor;
#endif
}

class DeepSeekV4KVCacheImpl final : public KVCacheImpl {
 public:
  explicit DeepSeekV4KVCacheImpl(const DeepSeekV4KVCacheTensors& tensors)
      : key_cache_(tensors.key_cache),
        index_cache_(tensors.index_cache),
        indexer_cache_scale_(tensors.indexer_cache_scale),
        swa_cache_(tensors.swa_cache),
        compress_kv_state_(tensors.compress_kv_state),
        compress_score_state_(tensors.compress_score_state),
        compress_index_kv_state_(tensors.compress_index_kv_state),
        compress_index_score_state_(tensors.compress_index_score_state) {}

  torch::Tensor get_k_cache() const override { return key_cache_; }

  torch::Tensor get_index_cache() const override { return index_cache_; }

  torch::Tensor get_indexer_cache_scale() const override {
    return indexer_cache_scale_;
  }

  torch::Tensor get_swa_cache() const override { return swa_cache_; }

  torch::Tensor get_compress_kv_state() const override {
    return compress_kv_state_;
  }

  torch::Tensor get_compress_score_state() const override {
    return compress_score_state_;
  }

  torch::Tensor get_compress_index_kv_state() const override {
    return compress_index_kv_state_;
  }

  torch::Tensor get_compress_index_score_state() const override {
    return compress_index_score_state_;
  }

  bool empty() const override { return !swa_cache_.defined(); }

  std::vector<std::vector<int64_t>> get_shapes() const override {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(3);
    if (key_cache_.defined()) {
      shapes.emplace_back(key_cache_.sizes().vec());
    }
    if (index_cache_.defined()) {
      shapes.emplace_back(index_cache_.sizes().vec());
    }
    if (swa_cache_.defined()) {
      shapes.emplace_back(swa_cache_.sizes().vec());
    }
    return shapes;
  }

  void swap_blocks(torch::Tensor& src_tensor,
                   torch::Tensor& dst_tensor) override {
    key_cache_ = swap_tensor_blocks(key_cache_, src_tensor, dst_tensor);
    index_cache_ = swap_tensor_blocks(index_cache_, src_tensor, dst_tensor);
    indexer_cache_scale_ =
        swap_tensor_blocks(indexer_cache_scale_, src_tensor, dst_tensor);
    swa_cache_ = swap_tensor_blocks(swa_cache_, src_tensor, dst_tensor);
    compress_kv_state_ =
        swap_tensor_blocks(compress_kv_state_, src_tensor, dst_tensor);
    compress_score_state_ =
        swap_tensor_blocks(compress_score_state_, src_tensor, dst_tensor);
    compress_index_kv_state_ =
        swap_tensor_blocks(compress_index_kv_state_, src_tensor, dst_tensor);
    compress_index_score_state_ =
        swap_tensor_blocks(compress_index_score_state_, src_tensor, dst_tensor);
  }

 private:
  torch::Tensor key_cache_;
  torch::Tensor index_cache_;
  torch::Tensor indexer_cache_scale_;
  torch::Tensor swa_cache_;
  torch::Tensor compress_kv_state_;
  torch::Tensor compress_score_state_;
  torch::Tensor compress_index_kv_state_;
  torch::Tensor compress_index_score_state_;
};

DeepSeekV4KVCacheTensors create_deepseek_v4_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_idx) {
  CHECK(kv_cache_shape.has_key_cache_shape())
      << "DeepSeek V4 cache shape must contain cache pool counts.";
  const std::vector<int64_t>& pool_counts = kv_cache_shape.key_cache_shape();
  CHECK_GE(pool_counts.size(), 3)
      << "DeepSeek V4 cache shape must be [swa_count, c4_count, c128_count].";
  CHECK_GT(create_options.block_size(), 0)
      << "DeepSeek V4 block_size must be positive.";
  CHECK_GT(create_options.head_dim(), 0)
      << "DeepSeek V4 head_dim must be positive.";
  CHECK_GT(create_options.window_size(), 0)
      << "DeepSeek V4 window_size must be positive.";

  const int64_t swa_count = pool_counts[0];
  const int64_t c4_count = pool_counts[1];
  const int64_t c128_count = pool_counts[2];
  const int64_t block_size = create_options.block_size();
  const int64_t head_dim = create_options.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(create_options.index_head_dim(), 1);
  const int64_t n_heads = 1;
  const int64_t index_n_heads = 1;
  const std::vector<int32_t>& compress_ratios =
      create_options.compress_ratios();
  const int32_t compress_ratio =
      layer_idx < static_cast<int64_t>(compress_ratios.size())
          ? compress_ratios[static_cast<size_t>(layer_idx)]
          : 1;

  const torch::TensorOptions cache_options =
      torch::dtype(create_options.dtype()).device(create_options.device());
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(create_options.dtype());
  const torch::TensorOptions index_options =
      torch::dtype(cache_policy.index_dtype).device(create_options.device());
  const torch::TensorOptions state_options =
      torch::dtype(torch::kFloat32).device(create_options.device());
  const torch::TensorOptions scale_options =
      torch::dtype(cache_policy.scale_dtype).device(create_options.device());

  DeepSeekV4KVCacheTensors tensors;
  if (compress_ratio == 1) {
    tensors.swa_cache =
        torch::empty({swa_count, block_size, n_heads, head_dim}, cache_options);
  } else if (compress_ratio == 4) {
    tensors.key_cache =
        torch::empty({c4_count, block_size, n_heads, head_dim}, cache_options);
    tensors.index_cache = torch::empty(
        {c4_count, block_size, index_n_heads, index_head_dim}, index_options);
    if (cache_policy.has_indexer_cache_scale) {
      tensors.indexer_cache_scale =
          torch::empty({c4_count, block_size, 1}, scale_options);
    }
    tensors.swa_cache =
        torch::empty({swa_count, block_size, n_heads, head_dim}, cache_options);
    tensors.compress_kv_state =
        torch::empty({swa_count, block_size, 2 * head_dim}, state_options);
    tensors.compress_score_state =
        torch::empty({swa_count, block_size, 2 * head_dim}, state_options);
    tensors.compress_index_kv_state = torch::empty(
        {swa_count, block_size, 2 * index_head_dim}, state_options);
    tensors.compress_index_score_state = torch::empty(
        {swa_count, block_size, 2 * index_head_dim}, state_options);
  } else if (compress_ratio == 128) {
    tensors.key_cache = torch::empty(
        {c128_count, block_size, n_heads, head_dim}, cache_options);
    tensors.swa_cache =
        torch::empty({swa_count, block_size, n_heads, head_dim}, cache_options);
    tensors.compress_kv_state =
        torch::empty({swa_count, block_size, head_dim}, state_options);
    tensors.compress_score_state =
        torch::empty({swa_count, block_size, head_dim}, state_options);
  } else {
    tensors.swa_cache =
        torch::empty({swa_count, block_size, n_heads, head_dim}, cache_options);
  }

  tensors.key_cache = cast_to_nd_format(tensors.key_cache);
  tensors.index_cache = cast_to_nd_format(tensors.index_cache);
  tensors.indexer_cache_scale = cast_to_nd_format(tensors.indexer_cache_scale);
  tensors.swa_cache = cast_to_nd_format(tensors.swa_cache);
  tensors.compress_kv_state = cast_to_nd_format(tensors.compress_kv_state);
  tensors.compress_score_state =
      cast_to_nd_format(tensors.compress_score_state);
  tensors.compress_index_kv_state =
      cast_to_nd_format(tensors.compress_index_kv_state);
  tensors.compress_index_score_state =
      cast_to_nd_format(tensors.compress_index_score_state);
  return tensors;
}

std::string deepseek_v4_shape_summary(const DeepSeekV4KVCacheTensors& tensors,
                                      int32_t compress_ratio) {
  std::ostringstream summary;
  if (compress_ratio == 1) {
    summary << "swa_cache=" << tensor_shape_string(tensors.swa_cache);
  } else if (compress_ratio == 4) {
    summary << "key_cache=" << tensor_shape_string(tensors.key_cache)
            << ", index_cache=" << tensor_shape_string(tensors.index_cache)
            << ", indexer_cache_scale="
            << tensor_shape_string(tensors.indexer_cache_scale)
            << ", swa_cache=" << tensor_shape_string(tensors.swa_cache)
            << ", compress_kv_state="
            << tensor_shape_string(tensors.compress_kv_state)
            << ", compress_score_state="
            << tensor_shape_string(tensors.compress_score_state)
            << ", compress_index_kv_state="
            << tensor_shape_string(tensors.compress_index_kv_state)
            << ", compress_index_score_state="
            << tensor_shape_string(tensors.compress_index_score_state);
  } else if (compress_ratio == 128) {
    summary << "key_cache=" << tensor_shape_string(tensors.key_cache)
            << ", swa_cache=" << tensor_shape_string(tensors.swa_cache)
            << ", compress_kv_state="
            << tensor_shape_string(tensors.compress_kv_state)
            << ", compress_score_state="
            << tensor_shape_string(tensors.compress_score_state);
  } else {
    summary << "swa_cache=" << tensor_shape_string(tensors.swa_cache);
  }
  return summary.str();
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

KVCache::KVCache(const DeepSeekV4KVCacheTensors& tensors)
    : impl_(std::make_unique<DeepSeekV4KVCacheImpl>(tensors)) {}

KVCache::KVCache(const KVCacheShape& kv_cache_shape,
                 const KVCacheCreateOptions& create_options,
                 int64_t layer_id)
    : impl_(create_kv_cache_impl(kv_cache_shape, create_options, layer_id)) {}

torch::Tensor KVCache::get_k_cache() const { return impl_->get_k_cache(); }

torch::Tensor KVCache::get_v_cache() const { return impl_->get_v_cache(); }

torch::Tensor KVCache::get_index_cache() const {
  return impl_->get_index_cache();
}

std::vector<KVCacheTensor> KVCache::get_cache_tensors() const {
  std::vector<KVCacheTensor> tensors;
  tensors.reserve(5);

  const torch::Tensor key_cache = get_k_cache();
  if (key_cache.defined() && key_cache.numel() > 0) {
    tensors.emplace_back(KVCacheTensorRole::KEY, key_cache);
  }

  const torch::Tensor value_cache = get_v_cache();
  if (value_cache.defined() && value_cache.numel() > 0) {
    tensors.emplace_back(KVCacheTensorRole::VALUE, value_cache);
  }

  const torch::Tensor index_cache = get_index_cache();
  if (index_cache.defined() && index_cache.numel() > 0) {
    tensors.emplace_back(KVCacheTensorRole::INDEX, index_cache);
  }

  const torch::Tensor conv_cache = get_conv_cache();
  if (conv_cache.defined() && conv_cache.numel() > 0) {
    tensors.emplace_back(KVCacheTensorRole::CONV, conv_cache);
  }

  const torch::Tensor ssm_cache = get_ssm_cache();
  if (ssm_cache.defined() && ssm_cache.numel() > 0) {
    tensors.emplace_back(KVCacheTensorRole::SSM, ssm_cache);
  }

  return tensors;
}

std::optional<torch::Tensor> KVCache::get_k_cache_scale() const {
  return impl_->get_k_cache_scale();
}

std::optional<torch::Tensor> KVCache::get_v_cache_scale() const {
  return impl_->get_v_cache_scale();
}

torch::Tensor KVCache::get_conv_cache() const {
  return impl_->get_conv_cache();
}

torch::Tensor KVCache::get_ssm_cache() const { return impl_->get_ssm_cache(); }

torch::Tensor KVCache::get_indexer_cache_scale() const {
  return impl_->get_indexer_cache_scale();
}

torch::Tensor KVCache::get_swa_cache() const { return impl_->get_swa_cache(); }

torch::Tensor KVCache::get_compress_kv_state() const {
  return impl_->get_compress_kv_state();
}

torch::Tensor KVCache::get_compress_score_state() const {
  return impl_->get_compress_score_state();
}

torch::Tensor KVCache::get_compress_index_kv_state() const {
  return impl_->get_compress_index_kv_state();
}

torch::Tensor KVCache::get_compress_index_score_state() const {
  return impl_->get_compress_index_score_state();
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

  if (create_options.model_type() == "deepseek_v4") {
    std::vector<int32_t> layer_compress_ratios;
    layer_compress_ratios.reserve(static_cast<size_t>(num_layers));
    std::map<int32_t, std::string> ratio_shape_summaries;
    const std::vector<int32_t>& compress_ratios =
        create_options.compress_ratios();

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      const int32_t compress_ratio =
          layer_idx < static_cast<int64_t>(compress_ratios.size())
              ? compress_ratios[static_cast<size_t>(layer_idx)]
              : 1;
      DeepSeekV4KVCacheTensors tensors = create_deepseek_v4_kv_cache_tensors(
          kv_cache_shape, create_options, layer_idx);
      layer_compress_ratios.emplace_back(compress_ratio);
      if (ratio_shape_summaries.find(compress_ratio) ==
          ratio_shape_summaries.end()) {
        ratio_shape_summaries.emplace(
            compress_ratio, deepseek_v4_shape_summary(tensors, compress_ratio));
      }
      kv_caches.emplace_back(tensors);
    }

    LOG(INFO) << "[DSV4][KVCacheInit] layer_crs: "
              << int32_vector_string(layer_compress_ratios);
    for (const std::pair<const int32_t, std::string>& summary :
         ratio_shape_summaries) {
      LOG(INFO) << "[DSV4][KVCacheInit] cr_" << summary.first
                << " shapes: " << summary.second;
    }
    return;
  }

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
