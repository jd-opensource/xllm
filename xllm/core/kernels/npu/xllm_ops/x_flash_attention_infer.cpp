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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "aclnn_x_flash_attention_infer.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace {

constexpr uint32_t kMaxExtraInfoNodes = 25;
constexpr uint32_t kMaxKvStackLen = 512;

struct CoreNode {
  uint32_t start_b_idx = 0;
  uint32_t start_n1_idx = 0;
  uint32_t start_s2_idx = 0;
  uint32_t end_b_idx = 0;
  uint32_t end_n1_idx = 0;
  uint32_t end_s2_idx = 0;
  uint64_t first_split_kv_task_lse_offset = 0;
  uint64_t first_split_kv_task_o_offset = 0;
};

struct SplitNode {
  uint32_t batch_idx = 0;
  uint32_t head_start_idx = 0;
  uint32_t head_end_idx = 0;
  uint32_t q_start_idx = 0;
  uint32_t q_end_idx = 0;
  uint32_t split_num = 0;
  uint64_t lse_task_offset = 0;
  uint64_t o_task_offset = 0;
};

struct SplitKvExtraInfo {
  CoreNode core_info[kMaxExtraInfoNodes];
  SplitNode split_info[kMaxExtraInfoNodes];
  uint32_t total_split_node_num = 0;
};

torch::Tensor make_extra_tiling(const torch::Tensor& actual_q_lens,
                                const torch::Tensor& actual_kv_lens,
                                int64_t q_head,
                                int64_t kv_head,
                                int64_t block_size,
                                int64_t head_size) {
  CHECK_EQ(actual_q_lens.dim(), 1) << "actual_q_lens must be 1-D";
  CHECK_EQ(actual_kv_lens.dim(), 1) << "actual_kv_lens must be 1-D";
  CHECK_EQ(actual_q_lens.numel(), actual_kv_lens.numel())
      << "actual_q_lens and actual_kv_lens must have the same length";
  CHECK_GT(actual_q_lens.numel(), 0) << "batch size must be positive";
  CHECK_GT(q_head, 0) << "q_head must be positive";
  CHECK_GT(kv_head, 0) << "kv_head must be positive";
  CHECK_EQ(q_head % kv_head, 0) << "q_head must be divisible by kv_head";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  CHECK_GT(head_size, 0) << "head_size must be positive";

  auto q_lens_cpu =
      actual_q_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  auto kv_lens_cpu =
      actual_kv_lens.to(torch::kCPU, torch::kInt32, /*non_blocking=*/false)
          .contiguous();
  const auto* q_lens = q_lens_cpu.data_ptr<int32_t>();
  const auto* kv_lens = kv_lens_cpu.data_ptr<int32_t>();
  const int64_t batch = actual_q_lens.numel();

  SplitKvExtraInfo extra_info;
  for (uint32_t i = 0; i < kMaxExtraInfoNodes; ++i) {
    extra_info.core_info[i].start_b_idx = std::numeric_limits<uint32_t>::max();
  }

  const uint32_t block_stack_num =
      std::max<uint32_t>(1, kMaxKvStackLen / static_cast<uint32_t>(block_size));
  const uint32_t group_size = static_cast<uint32_t>(q_head / kv_head);
  uint32_t core_idx = 0;
  uint32_t split_node_idx = 0;
  uint64_t lse_offset = 0;
  uint64_t o_offset = 0;

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const uint32_t q_end = static_cast<uint32_t>(q_lens[batch_idx]);
    const uint32_t q_start =
        batch_idx == 0 ? 0 : static_cast<uint32_t>(q_lens[batch_idx - 1]);
    const uint32_t q_len = q_end - q_start;
    const uint32_t kv_seq_len = static_cast<uint32_t>(kv_lens[batch_idx]);
    const uint32_t kv_blocks =
        (kv_seq_len + static_cast<uint32_t>(block_size) - 1) /
        static_cast<uint32_t>(block_size);
    const uint32_t s2_blocks =
        (kv_blocks + block_stack_num - 1) / block_stack_num;

    if (s2_blocks <= 1) {
      CHECK_LT(core_idx, kMaxExtraInfoNodes)
          << "x_flash_attention_infer extra_tiling core_info overflow";
      auto& core = extra_info.core_info[core_idx++];
      core.start_b_idx = static_cast<uint32_t>(batch_idx);
      core.start_n1_idx = 0;
      core.start_s2_idx = 0;
      core.end_b_idx = static_cast<uint32_t>(batch_idx);
      core.end_n1_idx = static_cast<uint32_t>(kv_head - 1);
      core.end_s2_idx = s2_blocks;
      continue;
    }

    // For long-KV requests, keep each KV head on one core and scan all S2
    // blocks inside that core. This avoids the split-KV combine path, whose
    // fixed coreInfo/splitInfo capacity is too small for long Qwen3.5 chunks.
    for (uint32_t kv_head_idx = 0; kv_head_idx < static_cast<uint32_t>(kv_head);
         ++kv_head_idx) {
      CHECK_LT(core_idx, kMaxExtraInfoNodes)
          << "x_flash_attention_infer extra_tiling core_info overflow";
      auto& core = extra_info.core_info[core_idx++];
      core.start_b_idx = static_cast<uint32_t>(batch_idx);
      core.start_n1_idx = kv_head_idx;
      core.start_s2_idx = 0;
      core.end_b_idx = static_cast<uint32_t>(batch_idx);
      core.end_n1_idx = kv_head_idx;
      core.end_s2_idx = s2_blocks;
    }
  }
  extra_info.total_split_node_num = split_node_idx;

  const int64_t int32_count =
      (sizeof(SplitKvExtraInfo) + sizeof(int32_t) - 1) / sizeof(int32_t);
  auto extra_tiling_cpu =
      torch::empty({int32_count}, torch::TensorOptions().dtype(torch::kInt32));
  std::memcpy(extra_tiling_cpu.data_ptr<int32_t>(),
              &extra_info,
              sizeof(SplitKvExtraInfo));
  return extra_tiling_cpu.to(actual_q_lens.device(), /*non_blocking=*/false);
}

}  // namespace

namespace xllm::kernel::npu {

torch::Tensor x_flash_attention_infer(const torch::Tensor& query,
                                      const torch::Tensor& key_cache,
                                      const torch::Tensor& value_cache,
                                      const torch::Tensor& mask,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& actual_q_lens,
                                      const torch::Tensor& actual_kv_lens,
                                      int64_t q_head,
                                      int64_t kv_head,
                                      double scale,
                                      const std::string& layout) {
  check_tensor(query, "query", "x_flash_attention_infer");
  check_tensor(key_cache, "key_cache", "x_flash_attention_infer");
  check_tensor(value_cache, "value_cache", "x_flash_attention_infer");
  check_tensor(mask, "mask", "x_flash_attention_infer");
  check_tensor(block_table, "block_table", "x_flash_attention_infer");
  check_tensor(actual_q_lens, "actual_q_lens", "x_flash_attention_infer");
  check_tensor(actual_kv_lens, "actual_kv_lens", "x_flash_attention_infer");
  CHECK_EQ(query.dim(), 3) << "query must be [tokens, q_heads, head_size]";
  CHECK_EQ(key_cache.dim(), 4)
      << "key_cache must be [blocks, block_size, kv_heads, head_size]";
  CHECK_EQ(value_cache.dim(), 4)
      << "value_cache must be [blocks, block_size, kv_heads, head_size]";
  CHECK_EQ(actual_q_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(actual_kv_lens.scalar_type(), torch::kInt32);

  torch::Tensor output = torch::empty_like(query);
  torch::Tensor extra_tiling = make_extra_tiling(actual_q_lens,
                                                 actual_kv_lens,
                                                 q_head,
                                                 kv_head,
                                                 key_cache.size(1),
                                                 key_cache.size(3));
  std::string layout_attr = layout;
  char* layout_attr_ptr = const_cast<char*>(layout_attr.c_str());

  EXEC_NPU_CMD(aclnnXFlashAttentionInfer,
               query,
               key_cache,
               value_cache,
               mask,
               block_table,
               actual_q_lens,
               actual_kv_lens,
               extra_tiling,
               layout_attr_ptr,
               q_head,
               kv_head,
               scale,
               output);
  return output;
}

}  // namespace xllm::kernel::npu
