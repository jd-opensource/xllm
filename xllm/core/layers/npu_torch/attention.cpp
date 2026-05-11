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

#include "attention.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

constexpr int64_t kFiaSplitFuseMaskSize = 2048;

std::vector<int64_t> cumulative_lengths(const std::vector<int32_t>& seq_lens) {
  std::vector<int64_t> cu_lens;
  cu_lens.reserve(seq_lens.size());
  int64_t total = 0;
  for (int32_t seq_len : seq_lens) {
    total += seq_len;
    cu_lens.emplace_back(total);
  }
  return cu_lens;
}

std::vector<int64_t> to_i64_vector(const std::vector<int32_t>& values) {
  std::vector<int64_t> out;
  out.reserve(values.size());
  for (int32_t value : values) {
    out.emplace_back(value);
  }
  return out;
}

torch::Tensor get_fia_split_fuse_attn_mask(const torch::Tensor& query) {
  static std::mutex mutex;
  static std::unordered_map<std::string, torch::Tensor> mask_cache;

  const std::string cache_key = query.device().str();
  std::lock_guard<std::mutex> lock(mutex);
  auto it = mask_cache.find(cache_key);
  if (it != mask_cache.end() && it->second.defined()) {
    return it->second;
  }

  auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  auto mask =
      torch::triu(torch::ones({kFiaSplitFuseMaskSize, kFiaSplitFuseMaskSize},
                              cpu_options),
                  1)
          .to(torch::kInt8)
          .to(query.device())
          .contiguous();
  mask_cache[cache_key] = mask;
  return mask;
}

}  // namespace

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        key,
        value,
        get_fia_split_fuse_attn_mask(query),
        std::nullopt,
        cumulative_lengths(attn_metadata.q_seq_lens_vec),
        cumulative_lengths(attn_metadata.kv_seq_lens_vec),
        num_heads_,
        num_kv_heads_,
        scale_,
        0,
        3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  } else if (attn_metadata.is_chunked_prefill) {
    auto q_seq_lens = cumulative_lengths(attn_metadata.q_seq_lens_vec);
    auto kv_seq_lens = to_i64_vector(attn_metadata.kv_seq_lens_vec);
    auto k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
    auto v = v_cache.value().view(
        {v_cache.value().size(0), v_cache.value().size(1), -1});
    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        k,
        v,
        get_fia_split_fuse_attn_mask(query),
        attn_metadata.block_table.defined()
            ? std::make_optional(attn_metadata.block_table)
            : std::nullopt,
        q_seq_lens,
        kv_seq_lens,
        num_heads_,
        num_kv_heads_,
        scale_,
        k_cache.size(1),
        3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (attn_metadata.paged_attention_tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(
        query,
        k_cache,
        v_cache.value_or(torch::Tensor()),
        scale_,
        attn_metadata.block_table,
        kv_seq_lens,
        attn_metadata.paged_attention_tiling_data,
        output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    attn_metadata.block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
