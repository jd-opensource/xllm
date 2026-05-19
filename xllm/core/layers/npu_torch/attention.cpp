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

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

constexpr int64_t kFiaSplitFuseMaskSize = 2048;
constexpr int64_t kXfaMaxQkRowsPerCall = 128;
constexpr int64_t kXfaMaxExtraInfoNodes = 24;
constexpr int64_t kXfaMaxKvStackLen = 512;

struct XfaQuerySlice {
  int64_t seq_idx = 0;
  int64_t q_start = 0;
  int64_t q_len = 0;
  int64_t kv_len = 0;
  int64_t core_count = 1;
};

torch::Tensor int32_tensor_on_device(const std::vector<int32_t>& values,
                                     const torch::Device& device) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
      .to(device);
}

torch::Tensor int64_tensor_on_device(const std::vector<int64_t>& values,
                                     const torch::Device& device) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64))
      .to(device);
}

int64_t div_up(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t xfa_core_count_for_slice(int64_t kv_len,
                                 int64_t kv_head,
                                 int64_t block_size) {
  const int64_t block_stack_num =
      std::max<int64_t>(1, kXfaMaxKvStackLen / block_size);
  const int64_t kv_blocks = div_up(kv_len, block_size);
  const int64_t s2_blocks = div_up(kv_blocks, block_stack_num);
  return s2_blocks <= 1 ? 1 : kv_head;
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

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
}

bool use_x_flash_decode() {
  return !env_flag_enabled("XLLM_DISABLE_XFLASH_DECODE");
}

torch::Tensor x_flash_attention_infer_with_query_split(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& mask,
    const torch::Tensor& block_table,
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& q_seq_lens_host,
    const torch::Tensor& kv_seq_lens_host,
    int64_t q_head,
    int64_t kv_head,
    double scale) {
  CHECK_GT(kv_head, 0);
  CHECK_EQ(q_head % kv_head, 0);
  const int64_t group_size = q_head / kv_head;
  const int64_t max_q_len_per_call =
      std::max<int64_t>(1, kXfaMaxQkRowsPerCall / group_size);

  auto q_seq_lens_cpu =
      q_seq_lens_host.defined()
          ? q_seq_lens_host
                .to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
                .contiguous()
          : q_seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
                .contiguous();
  auto kv_seq_lens_cpu =
      kv_seq_lens_host.defined()
          ? kv_seq_lens_host
                .to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
                .contiguous()
          : kv_seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
                .contiguous();
  CHECK_EQ(q_seq_lens_cpu.numel(), kv_seq_lens_cpu.numel());
  const auto* q_seq_lens_ptr = q_seq_lens_cpu.data_ptr<int32_t>();
  const auto* kv_seq_lens_ptr = kv_seq_lens_cpu.data_ptr<int32_t>();
  const int64_t batch = q_seq_lens_cpu.numel();

  auto output = torch::empty_like(query);
  int64_t q_offset = 0;
  std::vector<XfaQuerySlice> slices;
  const auto device = query.device();
  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t q_len = q_seq_lens_ptr[seq_idx];
    const int64_t kv_len = kv_seq_lens_ptr[seq_idx];
    const int64_t past_kv_len = kv_len - q_len;
    CHECK_GE(past_kv_len, 0);

    for (int64_t q_start = 0; q_start < q_len; q_start += max_q_len_per_call) {
      const int64_t sub_q_len =
          std::min<int64_t>(max_q_len_per_call, q_len - q_start);
      const int64_t sub_kv_len = past_kv_len + q_start + sub_q_len;
      const int64_t core_count =
          xfa_core_count_for_slice(sub_kv_len, kv_head, key_cache.size(1));
      CHECK_LE(core_count, kXfaMaxExtraInfoNodes)
          << "x_flash_attention_infer query slice needs too many cores";
      slices.push_back(
          {seq_idx, q_offset + q_start, sub_q_len, sub_kv_len, core_count});
    }
    q_offset += q_len;
  }

  for (size_t group_start = 0; group_start < slices.size();) {
    size_t group_end = group_start;
    int64_t core_count = 0;
    while (group_end < slices.size() &&
           core_count + slices[group_end].core_count <= kXfaMaxExtraInfoNodes) {
      core_count += slices[group_end].core_count;
      ++group_end;
    }

    const int64_t q_start = slices[group_start].q_start;
    const int64_t q_end =
        slices[group_end - 1].q_start + slices[group_end - 1].q_len;
    auto group_query = query.narrow(0, q_start, q_end - q_start).contiguous();

    std::vector<int32_t> group_q_lens;
    std::vector<int32_t> group_kv_lens;
    std::vector<int64_t> block_indices;
    group_q_lens.reserve(group_end - group_start);
    group_kv_lens.reserve(group_end - group_start);
    block_indices.reserve(group_end - group_start);
    int32_t q_cu_len = 0;
    for (size_t idx = group_start; idx < group_end; ++idx) {
      q_cu_len += static_cast<int32_t>(slices[idx].q_len);
      group_q_lens.push_back(q_cu_len);
      group_kv_lens.push_back(static_cast<int32_t>(slices[idx].kv_len));
      block_indices.push_back(slices[idx].seq_idx);
    }

    auto group_block_table =
        block_table
            .index_select(0, int64_tensor_on_device(block_indices, device))
            .contiguous();
    auto group_q_lens_tensor = int32_tensor_on_device(group_q_lens, device);
    auto group_kv_lens_tensor = int32_tensor_on_device(group_kv_lens, device);

    auto group_output =
        xllm::kernel::npu::x_flash_attention_infer(group_query,
                                                   key_cache,
                                                   value_cache,
                                                   mask,
                                                   group_block_table,
                                                   group_q_lens_tensor,
                                                   group_kv_lens_tensor,
                                                   q_head,
                                                   kv_head,
                                                   scale,
                                                   "TND");
    output.narrow(0, q_start, q_end - q_start).copy_(group_output);
    group_start = group_end;
  }
  return output;
}

torch::Tensor x_flash_attention_infer_decode_with_batch_split(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& mask,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    int64_t q_head,
    int64_t kv_head,
    double scale) {
  CHECK_EQ(query.dim(), 3)
      << "decode query must be [batch, q_heads, head_size]";
  auto kv_seq_lens_cpu =
      kv_seq_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  CHECK_EQ(query.size(0), kv_seq_lens_cpu.numel());
  const auto* kv_seq_lens_ptr = kv_seq_lens_cpu.data_ptr<int32_t>();
  const int64_t batch = query.size(0);
  const auto device = query.device();
  auto output = torch::empty_like(query);

  for (int64_t group_start = 0; group_start < batch;) {
    int64_t group_end = group_start;
    int64_t core_count = 0;
    while (group_end < batch) {
      const int64_t kv_len = kv_seq_lens_ptr[group_end];
      const int64_t seq_core_count =
          xfa_core_count_for_slice(kv_len, kv_head, key_cache.size(1));
      CHECK_LE(seq_core_count, kXfaMaxExtraInfoNodes)
          << "x_flash_attention_infer decode request needs too many cores";
      if (group_end > group_start &&
          core_count + seq_core_count > kXfaMaxExtraInfoNodes) {
        break;
      }
      core_count += seq_core_count;
      ++group_end;
    }

    const int64_t group_batch = group_end - group_start;
    std::vector<int32_t> group_q_lens;
    std::vector<int32_t> group_kv_lens;
    group_q_lens.reserve(group_batch);
    group_kv_lens.reserve(group_batch);
    int32_t q_cu_len = 0;
    for (int64_t idx = group_start; idx < group_end; ++idx) {
      ++q_cu_len;
      group_q_lens.push_back(q_cu_len);
      group_kv_lens.push_back(kv_seq_lens_ptr[idx]);
    }

    auto group_query = query.narrow(0, group_start, group_batch).contiguous();
    auto group_block_table =
        block_table.narrow(0, group_start, group_batch).contiguous();
    auto group_q_lens_tensor = int32_tensor_on_device(group_q_lens, device);
    auto group_kv_lens_tensor = int32_tensor_on_device(group_kv_lens, device);

    auto group_output =
        xllm::kernel::npu::x_flash_attention_infer(group_query,
                                                   key_cache,
                                                   value_cache,
                                                   mask,
                                                   group_block_table,
                                                   group_q_lens_tensor,
                                                   group_kv_lens_tensor,
                                                   q_head,
                                                   kv_head,
                                                   scale,
                                                   "TND");
    output.narrow(0, group_start, group_batch).copy_(group_output);
    group_start = group_end;
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

  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else if (only_prefill) {
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

  if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    auto xfa_result = x_flash_attention_infer_with_query_split(
        query,
        k_cache,
        v_cache.value(),
        get_fia_split_fuse_attn_mask(query),
        attn_metadata.block_table,
        attn_metadata.q_seq_lens,
        attn_metadata.kv_seq_lens,
        attn_metadata.q_seq_lens_host,
        attn_metadata.kv_seq_lens_host,
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
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor tiling_data = attn_metadata.paged_attention_tiling_data;
  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    block_table = attn_metadata.expanded_block_table;
    tiling_data = attn_metadata.expanded_paged_attention_tiling_data;
    if (attn_metadata.expanded_kv_seq_lens_host.defined()) {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens_host;
    } else {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens;
    }
  } else if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (!tiling_data.defined() && v_cache.has_value() && use_x_flash_decode()) {
    auto decode_query = query.view({-1, num_heads_, head_size_});
    auto xfa_result = x_flash_attention_infer_decode_with_batch_split(
        decode_query,
        k_cache,
        v_cache.value(),
        get_fia_split_fuse_attn_mask(decode_query),
        block_table,
        kv_seq_lens,
        num_heads_,
        num_kv_heads_,
        scale_);
    output.copy_(xfa_result.view_as(output));
    return;
  }

  if (tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(query,
                                              k_cache,
                                              v_cache.value_or(torch::Tensor()),
                                              scale_,
                                              block_table,
                                              kv_seq_lens,
                                              tiling_data,
                                              output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
