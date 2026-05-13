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
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

constexpr int64_t kFiaSplitFuseMaskSize = 2048;
constexpr int64_t kXfaMaxQkRowsPerCall = 128;

torch::Tensor cumulative_lengths_tensor(const torch::Tensor& seq_lens) {
  return torch::cumsum(seq_lens, 0).to(torch::kInt32);
}

std::vector<int64_t> cumulative_lengths(const torch::Tensor& seq_lens) {
  auto seq_lens_cpu =
      seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  const auto* seq_lens_ptr = seq_lens_cpu.data_ptr<int32_t>();
  std::vector<int64_t> cu_lens;
  cu_lens.reserve(seq_lens_cpu.numel());
  int64_t total = 0;
  for (int64_t i = 0; i < seq_lens_cpu.numel(); ++i) {
    total += seq_lens_ptr[i];
    cu_lens.emplace_back(total);
  }
  return cu_lens;
}

torch::Tensor int32_tensor_on_device(const std::vector<int32_t>& values,
                                     const torch::Device& device) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
      .to(device);
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

torch::Tensor x_flash_attention_infer_with_query_split(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& mask,
    const torch::Tensor& block_table,
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    int64_t q_head,
    int64_t kv_head,
    double scale) {
  CHECK_GT(kv_head, 0);
  CHECK_EQ(q_head % kv_head, 0);
  const int64_t group_size = q_head / kv_head;
  const int64_t max_q_len_per_call =
      std::max<int64_t>(1, kXfaMaxQkRowsPerCall / group_size);

  auto q_seq_lens_cpu =
      q_seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  auto kv_seq_lens_cpu =
      kv_seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  CHECK_EQ(q_seq_lens_cpu.numel(), kv_seq_lens_cpu.numel());
  const auto* q_seq_lens_ptr = q_seq_lens_cpu.data_ptr<int32_t>();
  const auto* kv_seq_lens_ptr = kv_seq_lens_cpu.data_ptr<int32_t>();
  const int64_t batch = q_seq_lens_cpu.numel();

  auto output = torch::empty_like(query);
  int64_t q_offset = 0;
  const auto device = query.device();
  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t q_len = q_seq_lens_ptr[seq_idx];
    const int64_t kv_len = kv_seq_lens_ptr[seq_idx];
    const int64_t past_kv_len = kv_len - q_len;
    CHECK_GE(past_kv_len, 0);

    for (int64_t q_start = 0; q_start < q_len; q_start += max_q_len_per_call) {
      const int64_t sub_q_len =
          std::min<int64_t>(max_q_len_per_call, q_len - q_start);
      auto sub_query =
          query.narrow(0, q_offset + q_start, sub_q_len).contiguous();
      auto sub_block_table = block_table.narrow(0, seq_idx, 1).contiguous();
      auto sub_q_lens =
          int32_tensor_on_device({static_cast<int32_t>(sub_q_len)}, device);
      auto sub_kv_lens = int32_tensor_on_device(
          {static_cast<int32_t>(past_kv_len + q_start + sub_q_len)}, device);

      auto sub_output =
          xllm::kernel::npu::x_flash_attention_infer(sub_query,
                                                     key_cache,
                                                     value_cache,
                                                     mask,
                                                     sub_block_table,
                                                     sub_q_lens,
                                                     sub_kv_lens,
                                                     q_head,
                                                     kv_head,
                                                     scale,
                                                     "TND");
      output.narrow(0, q_offset + q_start, sub_q_len).copy_(sub_output);
    }
    q_offset += q_len;
  }
  return output;
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
    auto xfa_result = x_flash_attention_infer_with_query_split(
        query,
        k_cache,
        v_cache.value(),
        get_fia_split_fuse_attn_mask(query),
        attn_metadata.block_table,
        attn_metadata.q_seq_lens,
        attn_metadata.kv_seq_lens,
        num_heads_,
        num_kv_heads_,
        scale_);
    output.copy_(xfa_result.view_as(output));
  } else if (attn_metadata.is_chunked_prefill) {
    auto xfa_result = x_flash_attention_infer_with_query_split(
        query,
        k_cache,
        v_cache.value(),
        get_fia_split_fuse_attn_mask(query),
        attn_metadata.block_table,
        attn_metadata.q_seq_lens,
        attn_metadata.kv_seq_lens,
        num_heads_,
        num_kv_heads_,
        scale_);
    output.copy_(xfa_result.view_as(output));
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
