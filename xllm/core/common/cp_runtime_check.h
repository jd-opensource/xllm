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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "framework/model/model_input_params.h"

#ifndef XLLM_CP_CHECK_LEVEL
#define XLLM_CP_CHECK_LEVEL 1
#endif

namespace xllm {
namespace cp_check {

inline std::vector<int64_t> XLLM_CPCHK_GET_Q_SEQ_LENS(
    const ModelInputParams& params) {
  std::vector<int64_t> q_seq_lens;
  const int32_t num_sequences = std::max(0, params.num_sequences);
  if (num_sequences == 0) {
    return q_seq_lens;
  }

  if (!params.q_seq_lens_vec.empty()) {
    const bool is_cumsum = params.q_seq_lens_vec.size() ==
                               static_cast<size_t>(num_sequences + 1) &&
                           params.q_seq_lens_vec.front() == 0;
    q_seq_lens.reserve(num_sequences);
    if (is_cumsum) {
      for (int32_t i = 0; i < num_sequences; ++i) {
        q_seq_lens.push_back(static_cast<int64_t>(params.q_seq_lens_vec[i + 1] -
                                                  params.q_seq_lens_vec[i]));
      }
      return q_seq_lens;
    }
    if (params.q_seq_lens_vec.size() == static_cast<size_t>(num_sequences)) {
      for (int32_t len : params.q_seq_lens_vec) {
        q_seq_lens.push_back(static_cast<int64_t>(len));
      }
      return q_seq_lens;
    }
  }

  if (params.q_seq_lens.defined() && params.q_seq_lens.dim() == 1 &&
      params.q_seq_lens.numel() > 0) {
    auto q_seq = params.q_seq_lens.to(torch::kCPU).to(torch::kLong).contiguous();
    const int64_t* ptr = q_seq.data_ptr<int64_t>();
    const int64_t n = q_seq.numel();
    const bool is_cumsum = n == num_sequences + 1 && ptr[0] == 0;
    q_seq_lens.reserve(num_sequences);
    if (is_cumsum) {
      for (int64_t i = 0; i < num_sequences; ++i) {
        q_seq_lens.push_back(ptr[i + 1] - ptr[i]);
      }
      return q_seq_lens;
    }
    if (n == num_sequences) {
      for (int64_t i = 0; i < n; ++i) {
        q_seq_lens.push_back(ptr[i]);
      }
      return q_seq_lens;
    }
  }

  return q_seq_lens;
}

inline int64_t XLLM_CPCHK_SUM(const std::vector<int64_t>& values) {
  return std::accumulate(values.begin(), values.end(), static_cast<int64_t>(0));
}

inline int32_t XLLM_CPCHK_CALC_TP_SIZE(int32_t world_size,
                                       int32_t dp_size,
                                       int32_t cp_size) {
  const int64_t denom = static_cast<int64_t>(dp_size) * cp_size;
  return (denom > 0) ? static_cast<int32_t>(world_size / denom) : 0;
}

inline int32_t XLLM_CPCHK_CALC_CP_RANK(int32_t rank,
                                       int32_t world_size,
                                       int32_t dp_size,
                                       int32_t cp_size) {
  const int32_t tp_size =
      std::max(1, XLLM_CPCHK_CALC_TP_SIZE(world_size, dp_size, cp_size));
  return (rank / tp_size) % std::max(1, cp_size);
}

inline void XLLM_CPCHK_CHECK_PARALLEL_ROLE(const char* stage,
                                           int32_t rank,
                                           int32_t world_size,
                                           int32_t dp_size,
                                           int32_t cp_size) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK_GT(world_size, 0) << "[XLLM_CPCHK][" << stage << "] world_size <= 0";
  CHECK_GT(dp_size, 0) << "[XLLM_CPCHK][" << stage << "] dp_size <= 0";
  CHECK_GT(cp_size, 0) << "[XLLM_CPCHK][" << stage << "] cp_size <= 0";
  CHECK_GE(rank, 0) << "[XLLM_CPCHK][" << stage << "] rank < 0";
  CHECK_LT(rank, world_size)
      << "[XLLM_CPCHK][" << stage << "] rank >= world_size";

  const int64_t prod = static_cast<int64_t>(dp_size) * cp_size;
  CHECK_EQ(world_size % prod, 0)
      << "[XLLM_CPCHK][" << stage
      << "] world_size must be divisible by dp_size * cp_size";

  const int32_t tp_size = XLLM_CPCHK_CALC_TP_SIZE(world_size, dp_size, cp_size);
  CHECK_GT(tp_size, 0) << "[XLLM_CPCHK][" << stage << "] tp_size <= 0";

  const int32_t dp_rank = rank / (tp_size * cp_size);
  const int32_t cp_rank = (rank / tp_size) % cp_size;
  const int32_t tp_rank = rank % tp_size;
  CHECK_GE(dp_rank, 0) << "[XLLM_CPCHK][" << stage << "] dp_rank < 0";
  CHECK_LT(dp_rank, dp_size)
      << "[XLLM_CPCHK][" << stage << "] dp_rank out of range";
  CHECK_GE(cp_rank, 0) << "[XLLM_CPCHK][" << stage << "] cp_rank < 0";
  CHECK_LT(cp_rank, cp_size)
      << "[XLLM_CPCHK][" << stage << "] cp_rank out of range";
  CHECK_GE(tp_rank, 0) << "[XLLM_CPCHK][" << stage << "] tp_rank < 0";
  CHECK_LT(tp_rank, tp_size)
      << "[XLLM_CPCHK][" << stage << "] tp_rank out of range";
#endif
}

inline void XLLM_CPCHK_CHECK_CP_RANK(const char* stage,
                                     int32_t cp_rank,
                                     int32_t rank,
                                     int32_t world_size,
                                     int32_t dp_size,
                                     int32_t cp_size) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK_GE(cp_rank, 0) << "[XLLM_CPCHK][" << stage << "] cp_rank < 0";
  CHECK_LT(cp_rank, std::max(1, cp_size))
      << "[XLLM_CPCHK][" << stage << "] cp_rank out of range";
  const int32_t expected =
      XLLM_CPCHK_CALC_CP_RANK(rank, world_size, dp_size, cp_size);
  CHECK_EQ(cp_rank, expected)
      << "[XLLM_CPCHK][" << stage << "] cp_rank mismatch";
#endif
}

inline void XLLM_CPCHK_CHECK_CP_PARTITION_BEFORE(
    const char* stage,
    const torch::Tensor& token_ids,
    const torch::Tensor& positions,
    const ModelInputParams& input_params,
    int32_t cp_rank,
    int32_t cp_size) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK_GT(cp_size, 1) << "[XLLM_CPCHK][" << stage << "] cp_size should be > 1";
  CHECK_GE(cp_rank, 0) << "[XLLM_CPCHK][" << stage << "] cp_rank < 0";
  CHECK_LT(cp_rank, cp_size)
      << "[XLLM_CPCHK][" << stage << "] cp_rank out of range";
  CHECK(token_ids.defined()) << "[XLLM_CPCHK][" << stage << "] token_ids undefined";
  CHECK_GT(token_ids.numel(), 0)
      << "[XLLM_CPCHK][" << stage << "] token_ids empty";

  const std::vector<int64_t> q_seq_lens = XLLM_CPCHK_GET_Q_SEQ_LENS(input_params);
  CHECK_EQ(q_seq_lens.size(), static_cast<size_t>(std::max(0, input_params.num_sequences)))
      << "[XLLM_CPCHK][" << stage << "] invalid q_seq_lens size";
  const int64_t total_tokens = XLLM_CPCHK_SUM(q_seq_lens);
  CHECK_EQ(total_tokens, token_ids.numel())
      << "[XLLM_CPCHK][" << stage
      << "] sum(q_seq_lens) != token_ids.numel";

  if (positions.defined()) {
    if (positions.dim() == 1) {
      CHECK_EQ(positions.size(0), token_ids.size(0))
          << "[XLLM_CPCHK][" << stage
          << "] positions.size(0) != token_ids.size(0)";
    } else if (positions.dim() == 2) {
      CHECK_EQ(positions.size(1), token_ids.size(0))
          << "[XLLM_CPCHK][" << stage
          << "] positions.size(1) != token_ids.size(0)";
    } else {
      CHECK(false) << "[XLLM_CPCHK][" << stage << "] unsupported positions dim";
    }
  }
#endif
}

inline void XLLM_CPCHK_CHECK_SELECTED_TOKEN_IDXES(const char* stage,
                                                  const torch::Tensor& idxes,
                                                  int64_t token_num,
                                                  const char* field_name) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  if (!idxes.defined()) {
    return;
  }
  CHECK_GE(token_num, 0) << "[XLLM_CPCHK][" << stage << "] token_num < 0";
  auto idx = idxes.to(torch::kCPU).to(torch::kLong).contiguous().view({-1});
  if (idx.numel() == 0) {
    return;
  }
  CHECK_GT(token_num, 0)
      << "[XLLM_CPCHK][" << stage << "] token_num should be > 0 when idx exists";
  const int64_t min_idx = idx.min().item<int64_t>();
  const int64_t max_idx = idx.max().item<int64_t>();
  CHECK_GE(min_idx, 0)
      << "[XLLM_CPCHK][" << stage << "] " << field_name << " has negative index";
  CHECK_LT(max_idx, token_num)
      << "[XLLM_CPCHK][" << stage << "] " << field_name << " out of range";
#endif
}

inline void XLLM_CPCHK_CHECK_CP_PARTITION_AFTER(
    const char* stage,
    const torch::Tensor& before_token_ids,
    const torch::Tensor& after_token_ids,
    const torch::Tensor& after_positions,
    const ModelInputParams& out_params) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK(before_token_ids.defined())
      << "[XLLM_CPCHK][" << stage << "] before_token_ids undefined";
  CHECK(after_token_ids.defined())
      << "[XLLM_CPCHK][" << stage << "] after_token_ids undefined";
  CHECK_LE(after_token_ids.numel(), before_token_ids.numel())
      << "[XLLM_CPCHK][" << stage << "] after token num > before token num";

  const std::vector<int64_t> q_seq_lens = XLLM_CPCHK_GET_Q_SEQ_LENS(out_params);
  CHECK_EQ(q_seq_lens.size(), static_cast<size_t>(std::max(0, out_params.num_sequences)))
      << "[XLLM_CPCHK][" << stage << "] invalid out q_seq_lens size";
  const int64_t local_tokens = XLLM_CPCHK_SUM(q_seq_lens);
  CHECK_EQ(local_tokens, after_token_ids.numel())
      << "[XLLM_CPCHK][" << stage
      << "] sum(out q_seq_lens) != after_token_ids.numel";

  if (after_positions.defined()) {
    if (after_positions.dim() == 1) {
      CHECK_EQ(after_positions.size(0), after_token_ids.size(0))
          << "[XLLM_CPCHK][" << stage
          << "] after positions.size(0) mismatch";
    } else if (after_positions.dim() == 2) {
      CHECK_EQ(after_positions.size(1), after_token_ids.size(0))
          << "[XLLM_CPCHK][" << stage
          << "] after positions.size(1) mismatch";
    } else {
      CHECK(false) << "[XLLM_CPCHK][" << stage << "] unsupported after positions dim";
    }
  }

  CHECK_EQ(out_params.q_max_seq_len, out_params.kv_max_seq_len)
      << "[XLLM_CPCHK][" << stage << "] q_max_seq_len != kv_max_seq_len";

  if (out_params.q_cu_seq_lens.defined()) {
    CHECK_EQ(out_params.q_cu_seq_lens.dim(), 1)
        << "[XLLM_CPCHK][" << stage << "] q_cu_seq_lens must be 1D";
    if (out_params.num_sequences > 0) {
      CHECK_EQ(out_params.q_cu_seq_lens.numel(), out_params.num_sequences)
          << "[XLLM_CPCHK][" << stage << "] q_cu_seq_lens numel mismatch";
      auto q_cu_seq_lens_cpu =
          out_params.q_cu_seq_lens.to(torch::kCPU).to(torch::kLong).contiguous();
      const int64_t last =
          q_cu_seq_lens_cpu.data_ptr<int64_t>()[q_cu_seq_lens_cpu.numel() - 1];
      CHECK_EQ(last, local_tokens)
          << "[XLLM_CPCHK][" << stage << "] q_cu_seq_lens last != local_tokens";
    }
  }
#endif
}

inline void XLLM_CPCHK_CHECK_TOKEN_LEVEL_TENSOR(
    const char* stage,
    const torch::Tensor& tensor,
    int64_t expected_tokens,
    const char* name) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  if (!tensor.defined()) {
    return;
  }
  CHECK_GE(tensor.dim(), 1)
      << "[XLLM_CPCHK][" << stage << "] " << name << " dim < 1";
  CHECK_EQ(tensor.size(0), expected_tokens)
      << "[XLLM_CPCHK][" << stage << "] " << name << " size(0) mismatch";
#endif
}

inline void XLLM_CPCHK_CHECK_TP_CP_SHARD_CONFIG(const char* stage,
                                                int32_t rank,
                                                int32_t world_size,
                                                int32_t dp_size,
                                                int32_t cp_size,
                                                int32_t dp_local_tp_size,
                                                int32_t dp_local_tp_rank) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  XLLM_CPCHK_CHECK_PARALLEL_ROLE(stage, rank, world_size, dp_size, cp_size);
  const int32_t expected_tp =
      XLLM_CPCHK_CALC_TP_SIZE(world_size, dp_size, cp_size);
  CHECK_EQ(dp_local_tp_size, expected_tp)
      << "[XLLM_CPCHK][" << stage
      << "] dp_local_tp_size mismatch";
  CHECK_GE(dp_local_tp_rank, 0)
      << "[XLLM_CPCHK][" << stage << "] dp_local_tp_rank < 0";
  CHECK_LT(dp_local_tp_rank, std::max(1, dp_local_tp_size))
      << "[XLLM_CPCHK][" << stage << "] dp_local_tp_rank out of range";
#endif
}

inline void XLLM_CPCHK_CHECK_KV_WEIGHT_SHAPE(const char* stage,
                                             const torch::Tensor& kv_weight,
                                             int32_t num_key_value_heads,
                                             int32_t dp_local_tp_size,
                                             int32_t qk_nope_head_dim,
                                             int32_t v_head_dim,
                                             int32_t kv_lora_rank) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK(kv_weight.defined()) << "[XLLM_CPCHK][" << stage << "] kv_weight undefined";
  CHECK_GT(dp_local_tp_size, 0)
      << "[XLLM_CPCHK][" << stage << "] dp_local_tp_size <= 0";
  CHECK_EQ(num_key_value_heads % dp_local_tp_size, 0)
      << "[XLLM_CPCHK][" << stage
      << "] num_key_value_heads not divisible by dp_local_tp_size";
  const int64_t expected_numel =
      static_cast<int64_t>(num_key_value_heads / dp_local_tp_size) *
      (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
  CHECK_EQ(kv_weight.numel(), expected_numel)
      << "[XLLM_CPCHK][" << stage << "] kv_weight numel mismatch";
#endif
}

inline void XLLM_CPCHK_CHECK_VARIANT_PACK_CAPACITY(const char* stage,
                                                   size_t variant_pack_size,
                                                   size_t required_size) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK_GE(variant_pack_size, required_size)
      << "[XLLM_CPCHK][" << stage
      << "] variantPack.inTensors capacity is insufficient";
#endif
}

inline void XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(const char* stage,
                                              const torch::Tensor& tensor,
                                              const char* name) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  CHECK(tensor.defined()) << "[XLLM_CPCHK][" << stage << "] " << name << " undefined";
  CHECK_EQ(tensor.dim(), 1)
      << "[XLLM_CPCHK][" << stage << "] " << name << " must be 1D";
  const auto dtype = tensor.scalar_type();
  CHECK(dtype == torch::kInt32 || dtype == torch::kInt64 || dtype == torch::kLong)
      << "[XLLM_CPCHK][" << stage << "] " << name << " dtype must be int32/int64";
#endif
}

inline void XLLM_CPCHK_CHECK_CP_PREFILL_TENSOR_SHAPES(
    const char* stage,
    const ModelInputParams& input_params,
    const torch::Tensor& seq_len_cp,
    const torch::Tensor& cp_load_balance_idx_first,
    const torch::Tensor& cp_load_balance_idx_last,
    const torch::Tensor& cp_o_recover_idx,
    const torch::Tensor& cp_kv_recover_idx,
    int32_t cp_size) {
#if defined(XLLM_ENABLE_CP_RUNTIME_CHECK)
  XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(stage, seq_len_cp, "seq_len_cp");
  XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(stage,
                                    cp_load_balance_idx_first,
                                    "cp_load_balance_idx_first");
  XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(stage,
                                    cp_load_balance_idx_last,
                                    "cp_load_balance_idx_last");
  XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(stage, cp_o_recover_idx, "cp_o_recover_idx");
  XLLM_CPCHK_CHECK_INT_INDEX_TENSOR(stage,
                                    cp_kv_recover_idx,
                                    "cp_kv_recover_idx");
  CHECK_GT(cp_size, 0) << "[XLLM_CPCHK][" << stage << "] cp_size <= 0";

  const std::vector<int64_t> q_seq_lens = XLLM_CPCHK_GET_Q_SEQ_LENS(input_params);
  const int64_t local_token_num = XLLM_CPCHK_SUM(q_seq_lens);
  CHECK_EQ(seq_len_cp.numel(), std::max<int64_t>(0, input_params.num_sequences))
      << "[XLLM_CPCHK][" << stage << "] seq_len_cp numel mismatch";

  const int64_t first_numel = cp_load_balance_idx_first.numel();
  const int64_t last_numel = cp_load_balance_idx_last.numel();
  CHECK_EQ(first_numel + last_numel, local_token_num)
      << "[XLLM_CPCHK][" << stage
      << "] load-balance index count mismatch";

  const int64_t chunk_sum =
      seq_len_cp.to(torch::kCPU).to(torch::kLong).contiguous().sum().item<int64_t>();
  CHECK_EQ(chunk_sum * 2, local_token_num)
      << "[XLLM_CPCHK][" << stage << "] seq_len_cp sum mismatch";

  CHECK_EQ(cp_o_recover_idx.numel(), local_token_num)
      << "[XLLM_CPCHK][" << stage << "] cp_o_recover_idx numel mismatch";
  CHECK_EQ(cp_kv_recover_idx.numel(), static_cast<int64_t>(cp_size) * local_token_num)
      << "[XLLM_CPCHK][" << stage << "] cp_kv_recover_idx numel mismatch";

#if XLLM_CP_CHECK_LEVEL >= 2
  if (local_token_num > 0) {
    auto first = cp_load_balance_idx_first.to(torch::kCPU).to(torch::kLong);
    auto last = cp_load_balance_idx_last.to(torch::kCPU).to(torch::kLong);
    auto o_rec = cp_o_recover_idx.to(torch::kCPU).to(torch::kLong);
    auto kv_rec = cp_kv_recover_idx.to(torch::kCPU).to(torch::kLong);
    if (first.numel() > 0) {
      CHECK_GE(first.min().item<int64_t>(), 0)
          << "[XLLM_CPCHK][" << stage << "] first idx < 0";
      CHECK_LT(first.max().item<int64_t>(), local_token_num)
          << "[XLLM_CPCHK][" << stage << "] first idx out of range";
    }
    if (last.numel() > 0) {
      CHECK_GE(last.min().item<int64_t>(), 0)
          << "[XLLM_CPCHK][" << stage << "] last idx < 0";
      CHECK_LT(last.max().item<int64_t>(), local_token_num)
          << "[XLLM_CPCHK][" << stage << "] last idx out of range";
    }
    CHECK_GE(o_rec.min().item<int64_t>(), 0)
        << "[XLLM_CPCHK][" << stage << "] o recover idx < 0";
    CHECK_LT(o_rec.max().item<int64_t>(), local_token_num)
        << "[XLLM_CPCHK][" << stage << "] o recover idx out of range";

    const int64_t kv_upper = static_cast<int64_t>(cp_size) * local_token_num;
    CHECK_GE(kv_rec.min().item<int64_t>(), 0)
        << "[XLLM_CPCHK][" << stage << "] kv recover idx < 0";
    CHECK_LT(kv_rec.max().item<int64_t>(), kv_upper)
        << "[XLLM_CPCHK][" << stage << "] kv recover idx out of range";
  }
#endif
#endif
}

}  // namespace cp_check
}  // namespace xllm
