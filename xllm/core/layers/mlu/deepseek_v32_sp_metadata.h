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
#include <numeric>
#include <vector>

#include "layers/common/attention_metadata.h"
#include "layers/mlu/deepseek_v32_sp_plan.h"

namespace xllm::layer::v32_sp {

struct DeepseekV32SPMetadata {
  std::vector<DeepseekV32SPSegment> segments;
  std::vector<int32_t> req_offsets_cpu;

  torch::Tensor seg_q_cu_lens;
  torch::Tensor seg_k_cu_lens;
  torch::Tensor seg_k_lens;
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
  std::vector<int32_t> seg_k_lens;
  seg_q_tokens.reserve(segments.size());
  seg_k_lens.reserve(segments.size());

  int32_t max_query_len = 0;
  int32_t max_seq_len = 0;
  for (const auto& segment : segments) {
    seg_q_tokens.push_back(segment.q_tokens);
    seg_k_lens.push_back(segment.k_len);
    max_query_len = std::max(max_query_len, segment.q_tokens);
    max_seq_len = std::max(max_seq_len, segment.k_len);
  }

  AttentionMetadata local_attn_metadata = base_attn_metadata;
  const torch::Device device = base_attn_metadata.q_cu_seq_lens.device();
  local_attn_metadata.q_cu_seq_lens =
      make_sp_prefix(seg_q_tokens, int32_options).to(device);
  local_attn_metadata.kv_cu_seq_lens =
      make_sp_prefix(seg_k_lens, int32_options).to(device);
  local_attn_metadata.kv_seq_lens =
      torch::tensor(seg_k_lens, int32_options).to(device);
  local_attn_metadata.max_query_len = max_query_len;
  local_attn_metadata.max_seq_len = max_seq_len;
  return local_attn_metadata;
}

inline DeepseekV32SPMetadata build_sp_metadata(
    const AttentionMetadata& base_attn_metadata,
    const std::vector<DeepseekV32SPSegment>& segments,
    const std::vector<int32_t>& seq_lens) {
  CHECK(base_attn_metadata.q_cu_seq_lens.defined())
      << "deepseek_v32 sequence parallel requires q_cu_seq_lens.";
  CHECK(base_attn_metadata.kv_seq_lens.defined())
      << "deepseek_v32 sequence parallel requires kv_seq_lens.";

  DeepseekV32SPMetadata meta;
  meta.segments = segments;

  meta.req_offsets_cpu.reserve(seq_lens.size());
  int32_t req_offset = 0;
  for (int32_t seq_len : seq_lens) {
    meta.req_offsets_cpu.push_back(req_offset);
    req_offset += seq_len;
  }

  auto q_cu_options = base_attn_metadata.q_cu_seq_lens.options();
  auto kv_seq_options = base_attn_metadata.kv_seq_lens.options();
  std::vector<int32_t> seg_q_tokens;
  std::vector<int32_t> seg_k_lens_cpu;
  std::vector<int64_t> seg_to_req_cpu;
  seg_q_tokens.reserve(meta.segments.size());
  seg_k_lens_cpu.reserve(meta.segments.size());
  seg_to_req_cpu.reserve(meta.segments.size());
  for (const auto& segment : meta.segments) {
    seg_q_tokens.push_back(segment.q_tokens);
    seg_k_lens_cpu.push_back(segment.k_len);
    seg_to_req_cpu.push_back(segment.req_idx);
  }
  meta.seg_q_cu_lens = make_sp_prefix(seg_q_tokens, q_cu_options);
  meta.seg_k_cu_lens = make_sp_prefix(seg_k_lens_cpu, q_cu_options);
  meta.seg_k_lens = torch::tensor(seg_k_lens_cpu, kv_seq_options);

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

inline torch::Tensor pack_sp_k_for_indexer(
    const torch::Tensor& k_global,
    const DeepseekV32SPMetadata& sp_meta) {
  if (!k_global.defined()) {
    return k_global;
  }
  CHECK_GT(k_global.dim(), 0)
      << "deepseek_v32 sequence parallel indexer expects non-scalar k.";

  std::vector<torch::Tensor> seg_slices;
  seg_slices.reserve(sp_meta.segments.size());
  for (const auto& segment : sp_meta.segments) {
    if (segment.k_len == 0) {
      continue;
    }
    const int32_t req_offset = sp_meta.req_offsets_cpu[segment.req_idx];
    seg_slices.push_back(k_global.narrow(0, req_offset, segment.k_len));
  }

  if (!seg_slices.empty()) {
    return torch::cat(seg_slices, 0).contiguous();
  }

  auto empty_shape = k_global.sizes().vec();
  empty_shape[0] = 0;
  return torch::empty(empty_shape, k_global.options());
}

}  // namespace xllm::layer::v32_sp
