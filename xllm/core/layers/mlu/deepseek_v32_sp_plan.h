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

namespace xllm::layer::v32_sp {

struct DeepseekV32SPSegment {
  int32_t req_idx = 0;
  int32_t rank = 0;
  int32_t q_tokens = 0;
  int32_t k_len = 0;
  int32_t world_begin = 0;
};

struct DeepseekV32SPCommPlan {
  std::vector<int32_t> tokens_per_rank;
  std::vector<int32_t> padded_tokens_per_rank;
  int32_t token_num_offset = 0;
};

struct DeepseekV32SPRuntimeArtifacts {
  DeepseekV32SPCommPlan comm_plan;
  std::vector<int32_t> gathered_reorder_index_cpu;
};

inline std::vector<int32_t> extract_prefill_seq_lens(
    const AttentionMetadata& attn_metadata) {
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "deepseek_v32 sequence parallel requires q_cu_seq_lens.";
  CHECK(attn_metadata.q_cu_seq_lens.dim() == 1)
      << "deepseek_v32 sequence parallel expects 1D q_cu_seq_lens.";
  CHECK(attn_metadata.q_cu_seq_lens.numel() >= 2)
      << "deepseek_v32 sequence parallel expects at least one request.";

  torch::Tensor q_cu_seq_lens = attn_metadata.q_cu_seq_lens.to(torch::kCPU)
                                    .to(torch::kInt64)
                                    .contiguous();
  const auto* q_cu_seq_lens_ptr = q_cu_seq_lens.data_ptr<int64_t>();

  std::vector<int32_t> seq_lens;
  seq_lens.reserve(q_cu_seq_lens.numel() - 1);
  for (int64_t i = 0; i < q_cu_seq_lens.numel() - 1; ++i) {
    const int64_t seq_len = q_cu_seq_lens_ptr[i + 1] - q_cu_seq_lens_ptr[i];
    CHECK_GE(seq_len, 0) << "q_cu_seq_lens must be non-decreasing.";
    seq_lens.push_back(static_cast<int32_t>(seq_len));
  }
  return seq_lens;
}

inline std::vector<int32_t> split_tokens_evenly(int32_t token_num,
                                                int32_t world_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive.";
  const int32_t base = token_num / world_size;
  const int32_t rem = token_num % world_size;

  std::vector<int32_t> token_num_split(world_size, base);
  for (int32_t i = 0; i < rem; ++i) {
    token_num_split[i] += 1;
  }
  return token_num_split;
}

inline std::vector<DeepseekV32SPSegment> build_all_sp_segments(
    int32_t world_size,
    const std::vector<int32_t>& seq_lens) {
  std::vector<DeepseekV32SPSegment> segments;
  segments.reserve(seq_lens.size() * world_size * 2);

  int32_t global_offset = 0;
  for (int32_t req_idx = 0; req_idx < seq_lens.size(); ++req_idx) {
    const int32_t token_num = seq_lens[req_idx];
    const std::vector<int32_t> split =
        split_tokens_evenly(token_num, world_size);

    int32_t batch_left = 0;
    int32_t batch_right = token_num;
    int32_t world_left = global_offset;
    int32_t world_right = global_offset + token_num;

    for (int32_t rank = 0; rank < world_size; ++rank) {
      const int32_t rank_token_num = split[rank];
      const int32_t left_token_num = rank_token_num / 2 + rank_token_num % 2;
      const int32_t right_token_num = rank_token_num / 2;

      DeepseekV32SPSegment left_segment;
      left_segment.req_idx = req_idx;
      left_segment.rank = rank;
      left_segment.q_tokens = left_token_num;
      left_segment.k_len =
          left_token_num == 0 ? 0 : batch_left + left_token_num;
      left_segment.world_begin = world_left;
      segments.push_back(left_segment);

      DeepseekV32SPSegment right_segment;
      right_segment.req_idx = req_idx;
      right_segment.rank = rank;
      right_segment.q_tokens = right_token_num;
      right_segment.k_len = right_token_num == 0 ? 0 : batch_right;
      right_segment.world_begin = world_right - right_token_num;
      segments.push_back(right_segment);

      world_left += left_token_num;
      world_right -= right_token_num;
      batch_left += left_token_num;
      batch_right -= right_token_num;
    }

    CHECK_EQ(batch_left, batch_right)
        << "zigzag split must fully consume request tokens.";
    CHECK_EQ(world_left, world_right) << "zigzag split world cursor mismatch.";
    global_offset += token_num;
  }
  return segments;
}

inline std::vector<DeepseekV32SPSegment> build_local_sp_segments(
    int32_t curr_rank,
    const std::vector<DeepseekV32SPSegment>& all_segments) {
  std::vector<DeepseekV32SPSegment> segments;
  for (const auto& segment : all_segments) {
    if (segment.rank == curr_rank) {
      segments.push_back(segment);
    }
  }
  return segments;
}

inline DeepseekV32SPRuntimeArtifacts build_sp_runtime_artifacts(
    int32_t curr_rank,
    int32_t world_size,
    const std::vector<DeepseekV32SPSegment>& all_segments,
    int32_t total_tokens) {
  CHECK_GE(curr_rank, 0) << "curr_rank must be non-negative.";
  CHECK_LT(curr_rank, world_size) << "curr_rank out of range.";

  DeepseekV32SPRuntimeArtifacts artifacts;
  auto& comm_plan = artifacts.comm_plan;
  comm_plan.tokens_per_rank.assign(world_size, 0);

  std::vector<std::vector<int32_t>> per_rank_indices(world_size);
  for (const auto& segment : all_segments) {
    comm_plan.tokens_per_rank[segment.rank] += segment.q_tokens;
    auto& rank_indices = per_rank_indices.at(segment.rank);
    if (rank_indices.empty()) {
      rank_indices.reserve(total_tokens / world_size + 1);
    }
    for (int32_t i = 0; i < segment.q_tokens; ++i) {
      rank_indices.push_back(segment.world_begin + i);
    }
  }

  const int32_t padded_token_num = *std::max_element(
      comm_plan.tokens_per_rank.begin(), comm_plan.tokens_per_rank.end());
  comm_plan.padded_tokens_per_rank.assign(world_size, padded_token_num);
  comm_plan.token_num_offset =
      std::accumulate(comm_plan.tokens_per_rank.begin(),
                      comm_plan.tokens_per_rank.begin() + curr_rank,
                      int32_t{0});

  artifacts.gathered_reorder_index_cpu.reserve(total_tokens);
  for (const auto& rank_indices : per_rank_indices) {
    artifacts.gathered_reorder_index_cpu.insert(
        artifacts.gathered_reorder_index_cpu.end(),
        rank_indices.begin(),
        rank_indices.end());
  }
  CHECK_EQ(artifacts.gathered_reorder_index_cpu.size(), total_tokens)
      << "gathered reorder index must cover all tokens.";
  return artifacts;
}

}  // namespace xllm::layer::v32_sp
