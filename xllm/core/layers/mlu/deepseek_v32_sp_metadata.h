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

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "layers/common/attention_metadata.h"
#include "layers/mlu/deepseek_v32_sp_plan.h"

namespace xllm::layer::v32_sp {

struct DeepseekV32SPMetadata {
  std::vector<int32_t> k_pack_starts_cpu;
  std::vector<int32_t> k_pack_lens_cpu;
  std::vector<int32_t> k_ctx_pack_starts_cpu;
  std::vector<int32_t> k_ctx_pack_lens_cpu;

  torch::Tensor seg_q_cu_lens;
  torch::Tensor seg_suffix_k_cu_lens;
  torch::Tensor seg_ctx_cu_lens;
  torch::Tensor seg_ctx_lens;
  torch::Tensor seg_block_table;
};

inline torch::Tensor make_sp_prefix(const std::vector<int32_t>& seq_lens,
                                    const torch::TensorOptions& options) {
  std::vector<int32_t> cu_lens = {0};
  cu_lens.reserve(seq_lens.size() + 1);
  int32_t token_num = 0;
  for (int32_t seq_len : seq_lens) {
    token_num += seq_len;
    cu_lens.push_back(token_num);
  }
  return torch::tensor(cu_lens, options);
}

inline AttentionMetadata build_local_prefill_attention_metadata(
    const AttentionMetadata& base_attn_metadata,
    const std::vector<DeepseekV32SPSegment>& segments) {
  const auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  std::vector<int32_t> seg_q_tokens;
  std::vector<int32_t> seg_suffix_k_lens;
  seg_q_tokens.reserve(segments.size());
  seg_suffix_k_lens.reserve(segments.size());

  int32_t max_query_len = 0;
  int32_t max_seq_len = 0;
  for (const auto& segment : segments) {
    CHECK_LE(segment.suffix_k_len, segment.ctx_k_len)
        << "deepseek_v32 sequence parallel expects suffix_k_len <= ctx_k_len.";
    seg_q_tokens.push_back(segment.q_tokens);
    seg_suffix_k_lens.push_back(segment.suffix_k_len);
    max_query_len = std::max(max_query_len, segment.q_tokens);
    max_seq_len = std::max(max_seq_len, segment.suffix_k_len);
  }

  AttentionMetadata local_attn_metadata = base_attn_metadata;
  const torch::Device device = base_attn_metadata.q_cu_seq_lens.device();
  local_attn_metadata.q_cu_seq_lens =
      make_sp_prefix(seg_q_tokens, int32_options).to(device);
  local_attn_metadata.kv_cu_seq_lens =
      make_sp_prefix(seg_suffix_k_lens, int32_options).to(device);
  // Local SP metadata stays on the live suffix view. Cached prefix tokens are
  // addressed later through seg_ctx_lens + block_table in the indexer path.
  local_attn_metadata.kv_seq_lens =
      torch::tensor(seg_suffix_k_lens, int32_options).to(device);
  local_attn_metadata.max_query_len = max_query_len;
  local_attn_metadata.max_seq_len = max_seq_len;
  return local_attn_metadata;
}

inline DeepseekV32SPMetadata build_sp_metadata(
    const AttentionMetadata& base_attn_metadata,
    const std::vector<DeepseekV32SPSegment>& segments,
    const std::vector<int32_t>& q_seq_lens) {
  CHECK(base_attn_metadata.q_cu_seq_lens.defined())
      << "deepseek_v32 sequence parallel requires q_cu_seq_lens.";
  CHECK(base_attn_metadata.kv_seq_lens.defined())
      << "deepseek_v32 sequence parallel requires kv_seq_lens.";

  DeepseekV32SPMetadata meta;
  std::vector<int32_t> req_offsets_cpu;
  req_offsets_cpu.reserve(q_seq_lens.size());
  int32_t req_offset = 0;
  for (int32_t q_seq_len : q_seq_lens) {
    req_offsets_cpu.push_back(req_offset);
    req_offset += q_seq_len;
  }

  auto q_cu_options = base_attn_metadata.q_cu_seq_lens.options();
  auto kv_seq_options = base_attn_metadata.kv_seq_lens.options();
  torch::Tensor kv_seq_lens_cpu = base_attn_metadata.kv_seq_lens.to(torch::kCPU)
                                      .to(torch::kInt64)
                                      .contiguous();
  const auto* kv_seq_lens_ptr = kv_seq_lens_cpu.data_ptr<int64_t>();
  std::vector<int32_t> req_ctx_offsets_cpu;
  req_ctx_offsets_cpu.reserve(kv_seq_lens_cpu.numel());
  int32_t req_ctx_offset = 0;
  for (int64_t i = 0; i < kv_seq_lens_cpu.numel(); ++i) {
    req_ctx_offsets_cpu.push_back(req_ctx_offset);
    req_ctx_offset += static_cast<int32_t>(kv_seq_lens_ptr[i]);
  }
  std::vector<int32_t> seg_q_tokens;
  std::vector<int32_t> seg_suffix_k_lens_cpu;
  std::vector<int32_t> seg_ctx_lens_cpu;
  std::vector<int64_t> seg_to_req_cpu;
  seg_q_tokens.reserve(segments.size());
  seg_suffix_k_lens_cpu.reserve(segments.size());
  seg_ctx_lens_cpu.reserve(segments.size());
  seg_to_req_cpu.reserve(segments.size());
  meta.k_pack_starts_cpu.reserve(segments.size());
  meta.k_pack_lens_cpu.reserve(segments.size());
  meta.k_ctx_pack_starts_cpu.reserve(segments.size());
  meta.k_ctx_pack_lens_cpu.reserve(segments.size());
  for (const auto& segment : segments) {
    seg_q_tokens.push_back(segment.q_tokens);
    seg_suffix_k_lens_cpu.push_back(segment.suffix_k_len);
    seg_ctx_lens_cpu.push_back(segment.ctx_k_len);
    seg_to_req_cpu.push_back(segment.req_idx);
    if (segment.suffix_k_len > 0) {
      meta.k_pack_starts_cpu.push_back(req_offsets_cpu[segment.req_idx]);
      meta.k_pack_lens_cpu.push_back(segment.suffix_k_len);
    }
    if (segment.ctx_k_len > 0) {
      meta.k_ctx_pack_starts_cpu.push_back(
          req_ctx_offsets_cpu[segment.req_idx]);
      meta.k_ctx_pack_lens_cpu.push_back(segment.ctx_k_len);
    }
  }
  meta.seg_q_cu_lens = make_sp_prefix(seg_q_tokens, q_cu_options);
  meta.seg_suffix_k_cu_lens =
      make_sp_prefix(seg_suffix_k_lens_cpu, q_cu_options);
  meta.seg_ctx_cu_lens = make_sp_prefix(seg_ctx_lens_cpu, q_cu_options);
  meta.seg_ctx_lens = torch::tensor(seg_ctx_lens_cpu, kv_seq_options);

  if (base_attn_metadata.block_table.defined()) {
    auto seg_to_req_index =
        torch::tensor(seg_to_req_cpu,
                      torch::TensorOptions()
                          .dtype(torch::kInt64)
                          .device(base_attn_metadata.block_table.device()));
    meta.seg_block_table =
        base_attn_metadata.block_table.index_select(0, seg_to_req_index);
  }
  return meta;
}

inline torch::Tensor pack_sp_k_slices(const torch::Tensor& k,
                                      const std::vector<int32_t>& starts_cpu,
                                      const std::vector<int32_t>& lens_cpu) {
  if (!k.defined()) {
    return k;
  }
  CHECK_GT(k.dim(), 0)
      << "deepseek_v32 sequence parallel indexer expects non-scalar k.";
  CHECK_EQ(starts_cpu.size(), lens_cpu.size())
      << "deepseek_v32 sequence parallel indexer expects aligned K slice "
         "starts/lens metadata.";

  std::vector<torch::Tensor> seg_slices;
  seg_slices.reserve(lens_cpu.size());
  for (size_t i = 0; i < lens_cpu.size(); ++i) {
    CHECK_GE(starts_cpu[i], 0)
        << "deepseek_v32 sequence parallel indexer expects non-negative K "
           "slice starts.";
    CHECK_GE(lens_cpu[i], 0)
        << "deepseek_v32 sequence parallel indexer expects non-negative K "
           "slice lengths.";
    CHECK_LE(static_cast<int64_t>(starts_cpu[i]) + lens_cpu[i], k.size(0))
        << "deepseek_v32 sequence parallel indexer K slice exceeds packed K "
           "rows.";
    seg_slices.push_back(k.narrow(0, starts_cpu[i], lens_cpu[i]));
  }

  if (!seg_slices.empty()) {
    return torch::cat(seg_slices, 0).contiguous();
  }

  auto empty_shape = k.sizes().vec();
  empty_shape[0] = 0;
  return torch::empty(empty_shape, k.options());
}

inline torch::Tensor pack_sp_k_for_indexer(
    const torch::Tensor& k_global,
    const DeepseekV32SPMetadata& sp_meta) {
  return pack_sp_k_slices(
      k_global, sp_meta.k_pack_starts_cpu, sp_meta.k_pack_lens_cpu);
}

inline torch::Tensor pack_sp_ctx_k(const torch::Tensor& k_ctx,
                                   const DeepseekV32SPMetadata& sp_meta) {
  return pack_sp_k_slices(
      k_ctx, sp_meta.k_ctx_pack_starts_cpu, sp_meta.k_ctx_pack_lens_cpu);
}

}  // namespace xllm::layer::v32_sp
