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

#include "runtime/profile_run.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#if defined(USE_MLU)
#pragma push_macro("USE_MLU")
#undef USE_MLU
#include "runtime/forward_params.h"
#pragma pop_macro("USE_MLU")
#else
#include "runtime/forward_params.h"
#endif
#include "framework/model/model_args.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace runtime {
namespace {

int32_t get_seqs(const Options& opt, int32_t num_tokens) {
  return std::max<int32_t>(1, std::min(opt.max_seqs_per_batch(), num_tokens));
}

std::vector<int32_t> split_tokens(int32_t num_tokens, int32_t num_seqs) {
  std::vector<int32_t> seq_tokens;
  seq_tokens.reserve(static_cast<size_t>(num_seqs));
  const int32_t base = num_tokens / num_seqs;
  const int32_t rem = num_tokens % num_seqs;
  for (int32_t i = 0; i < num_seqs; ++i) {
    seq_tokens.push_back(base + (i < rem ? 1 : 0));
  }
  return seq_tokens;
}

std::vector<int32_t> build_seq_lens(const std::vector<int32_t>& seq_tokens,
                                    bool is_mlu_build) {
  if (!is_mlu_build) {
    return seq_tokens;
  }

  std::vector<int32_t> seq_lens;
  seq_lens.reserve(seq_tokens.size() + 1);
  seq_lens.push_back(0);
  for (int32_t seq_len : seq_tokens) {
    seq_lens.push_back(seq_lens.back() + seq_len);
  }
  return seq_lens;
}

std::vector<std::vector<int32_t>> build_block_tables(
    const std::vector<int32_t>& seq_tokens,
    int32_t block_size) {
  std::vector<std::vector<int32_t>> block_tables;
  block_tables.reserve(seq_tokens.size());
  int32_t block_id = 0;
  for (int32_t seq_len : seq_tokens) {
    const int32_t num_blocks = (seq_len + block_size - 1) / block_size;
    std::vector<int32_t> row;
    row.reserve(static_cast<size_t>(num_blocks));
    for (int32_t j = 0; j < num_blocks; ++j) {
      row.push_back(block_id++);
    }
    block_tables.push_back(std::move(row));
  }
  return block_tables;
}

std::vector<int32_t> build_cu_lens(const std::vector<int32_t>& seq_lens) {
  std::vector<int32_t> cu_lens;
  cu_lens.reserve(seq_lens.size() + 1);
  cu_lens.push_back(0);
  for (int32_t seq_len : seq_lens) {
    cu_lens.push_back(cu_lens.back() + seq_len);
  }
  return cu_lens;
}

std::vector<int32_t> get_seq_lens(const ProfilePlan& plan, bool is_mlu_build) {
  if (!is_mlu_build) {
    return plan.seq_lens;
  }

  std::vector<int32_t> seq_lens;
  if (plan.seq_lens.size() <= 1) {
    return seq_lens;
  }
  seq_lens.reserve(plan.seq_lens.size() - 1);
  for (size_t i = 1; i < plan.seq_lens.size(); ++i) {
    seq_lens.push_back(plan.seq_lens[i] - plan.seq_lens[i - 1]);
  }
  return seq_lens;
}

std::vector<int32_t> build_positions(const std::vector<int32_t>& seq_lens) {
  std::vector<int32_t> positions;
  int64_t total_tokens = 0;
  for (int32_t seq_len : seq_lens) {
    total_tokens += seq_len;
  }
  positions.reserve(static_cast<size_t>(total_tokens));
  for (int32_t seq_len : seq_lens) {
    for (int32_t i = 0; i < seq_len; ++i) {
      positions.push_back(i);
    }
  }
  return positions;
}

std::vector<int32_t> build_new_slots(
    const std::vector<int32_t>& seq_lens,
    const std::vector<std::vector<int32_t>>& block_tables,
    int32_t block_size) {
  std::vector<int32_t> new_cache_slots;
  int64_t total_tokens = 0;
  for (int32_t seq_len : seq_lens) {
    total_tokens += seq_len;
  }
  new_cache_slots.reserve(static_cast<size_t>(total_tokens));

  for (size_t seq_idx = 0; seq_idx < seq_lens.size(); ++seq_idx) {
    const int32_t seq_len = seq_lens[seq_idx];
    const auto& row = block_tables[seq_idx];
    for (int32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      const int32_t block_idx = token_idx / block_size;
      const int32_t block_id = row[block_idx];
      new_cache_slots.push_back(block_id * block_size + token_idx % block_size);
    }
  }
  return new_cache_slots;
}

std::vector<int32_t> build_paged_indptr(const std::vector<int32_t>& seq_lens,
                                        int32_t block_size) {
  std::vector<int32_t> paged_kv_indptr;
  paged_kv_indptr.reserve(seq_lens.size() + 1);
  paged_kv_indptr.push_back(0);
  for (int32_t seq_len : seq_lens) {
    const int32_t num_blocks = (seq_len + block_size - 1) / block_size;
    paged_kv_indptr.push_back(paged_kv_indptr.back() + num_blocks);
  }
  return paged_kv_indptr;
}

std::vector<int32_t> build_paged_indices(
    const std::vector<std::vector<int32_t>>& block_tables) {
  std::vector<int32_t> paged_kv_indices;
  int64_t total_blocks = 0;
  for (const auto& row : block_tables) {
    total_blocks += row.size();
  }
  paged_kv_indices.reserve(static_cast<size_t>(total_blocks));
  for (const auto& row : block_tables) {
    paged_kv_indices.insert(paged_kv_indices.end(), row.begin(), row.end());
  }
  return paged_kv_indices;
}

std::vector<int32_t> build_paged_last_page(const std::vector<int32_t>& seq_lens,
                                           int32_t block_size) {
  std::vector<int32_t> paged_kv_last_page_len;
  paged_kv_last_page_len.reserve(seq_lens.size());
  for (int32_t seq_len : seq_lens) {
    const int32_t last_page_len = seq_len % block_size;
    paged_kv_last_page_len.push_back(last_page_len == 0 ? block_size
                                                        : last_page_len);
  }
  return paged_kv_last_page_len;
}

int64_t get_elem_bytes(const std::string& dtype) {
  if (dtype == "float32" || dtype == "fp32" || dtype == "f32") {
    return 4;
  }
  if (dtype == "int8" || dtype == "i8") {
    return 1;
  }
  return 2;
}

int64_t calc_tmp_kv_bytes(const ModelArgs& args,
                          const std::vector<int32_t>& seq_tokens,
                          int32_t block_size,
                          bool is_mla) {
  const int64_t elem_bytes = get_elem_bytes(args.dtype());
  const int64_t num_layers = args.n_layers();
  if (num_layers <= 0 || seq_tokens.empty()) {
    return 0;
  }
  int64_t num_blocks = 0;
  for (int32_t seq_len : seq_tokens) {
    num_blocks += (seq_len + block_size - 1) / block_size;
  }
  if (is_mla) {
    const int64_t kv_dim =
        std::max<int64_t>(0, args.kv_lora_rank() + args.qk_rope_head_dim());
    return num_layers * num_blocks * block_size * kv_dim * elem_bytes;
  }
  const int64_t n_kv_heads = args.n_kv_heads().has_value()
                                 ? args.n_kv_heads().value()
                                 : args.n_heads();
  return num_layers * num_blocks * block_size * n_kv_heads * args.head_dim() *
         elem_bytes * 2;
}

}  // namespace

bool use_profile_run(const Options& opt, bool is_mlu_build) {
  return opt.enable_profile_run() && is_mlu_build && !opt.enable_disagg_pd();
}

int32_t pick_profile_tokens(const Options& opt) {
  if (!opt.enable_chunked_prefill()) {
    return opt.max_tokens_per_batch();
  }
  const int32_t chunk_tokens = opt.max_tokens_per_chunk_for_prefill();
  if (chunk_tokens <= 0) {
    return opt.max_tokens_per_batch();
  }
  return std::min(opt.max_tokens_per_batch(), chunk_tokens);
}

int64_t calc_runtime_peak(const PeakMem& base, const PeakMem& peak) {
  const int64_t alloc_delta = peak.alloc_bytes - base.alloc_bytes;
  const int64_t cache_delta = peak.cache_bytes - base.cache_bytes;
  return std::max<int64_t>(0, std::max(alloc_delta, cache_delta));
}

int64_t calc_safe_kv_bytes(const ProfileMem& mem, const Options& opt) {
  if (!mem.ok || mem.total_bytes <= 0) {
    return 0;
  }
  const int64_t usable_bytes = static_cast<int64_t>(
      static_cast<double>(mem.total_bytes) * opt.max_memory_utilization());
  const int64_t safe_kv_bytes =
      usable_bytes - mem.weight_bytes - mem.runtime_peak_bytes;
  return std::max<int64_t>(0, safe_kv_bytes);
}

int64_t calc_safe_kv_bytes(const std::vector<ProfileMem>& worker_mems,
                           const Options& opt) {
  if (worker_mems.empty()) {
    return 0;
  }

  int64_t safe_kv_bytes = std::numeric_limits<int64_t>::max();
  for (const auto& mem : worker_mems) {
    safe_kv_bytes = std::min(safe_kv_bytes, calc_safe_kv_bytes(mem, opt));
  }

  if (opt.max_cache_size() > 0) {
    safe_kv_bytes = std::min(safe_kv_bytes, opt.max_cache_size());
  }

  if (safe_kv_bytes == std::numeric_limits<int64_t>::max()) {
    return 0;
  }
  return safe_kv_bytes;
}

ProfilePlan build_profile_plan(const ModelArgs& args,
                               const Options& opt,
                               int32_t block_size,
                               bool is_mla,
                               bool is_mlu_build) {
  CHECK_GT(block_size, 0);
  ProfilePlan plan;
  plan.num_tokens = pick_profile_tokens(opt);
  plan.num_seqs = get_seqs(opt, plan.num_tokens);
  const std::vector<int32_t> seq_tokens =
      split_tokens(plan.num_tokens, plan.num_seqs);
  plan.seq_lens = build_seq_lens(seq_tokens, is_mlu_build);
  plan.block_tables = build_block_tables(seq_tokens, block_size);
  plan.tmp_kv_bytes = calc_tmp_kv_bytes(args, seq_tokens, block_size, is_mla);
  return plan;
}

ForwardInput build_profile_input(const ModelArgs& args,
                                 const Options& opt,
                                 const ProfilePlan& plan,
                                 bool is_mlu_build) {
  ForwardInput input;
  const std::vector<int32_t> seq_lens = get_seq_lens(plan, is_mlu_build);
  const std::vector<int32_t> cu_lens = build_cu_lens(seq_lens);
  std::vector<std::vector<int32_t>> block_tables = plan.block_tables;
  util::pad_2d_vector(block_tables, /*pad_value=*/0);

  const int32_t block_size = opt.block_size();
  const std::vector<int32_t> positions = build_positions(seq_lens);
  const std::vector<int32_t> new_cache_slots =
      build_new_slots(seq_lens, plan.block_tables, block_size);
  const std::vector<int32_t> paged_kv_indptr =
      build_paged_indptr(seq_lens, block_size);
  const std::vector<int32_t> paged_kv_indices =
      build_paged_indices(plan.block_tables);
  const std::vector<int32_t> paged_kv_last_page_len =
      build_paged_last_page(seq_lens, block_size);
  const std::vector<int32_t> kv_cache_tokens_nums(seq_lens.size(), 0);
  std::vector<int32_t> token_ids(plan.num_tokens, 0);

  if (args.rope_scaling_mrope_section().empty()) {
    input.positions = torch::tensor(positions, torch::kInt);
  } else {
    input.positions =
        torch::zeros({3, static_cast<int64_t>(plan.num_tokens)}, torch::kInt);
  }
  input.token_ids = torch::tensor(token_ids, torch::kInt);

  auto& input_params = input.input_params;
  input_params.batch_forward_type = BatchForwardType::PREFILL;
  input_params.enable_mla = args.enable_mla();
  input_params.num_sequences = static_cast<int32_t>(seq_lens.size());
  input_params.kv_max_seq_len =
      seq_lens.empty() ? 0
                       : *std::max_element(seq_lens.begin(), seq_lens.end());
  input_params.q_max_seq_len = input_params.kv_max_seq_len;
  input_params.kv_seq_lens = torch::tensor(cu_lens, torch::kInt);
  input_params.q_seq_lens = torch::tensor(cu_lens, torch::kInt);
  input_params.q_cu_seq_lens = input_params.q_seq_lens;
  input_params.kv_seq_lens_vec = cu_lens;
  input_params.q_seq_lens_vec = cu_lens;
  input_params.kv_cache_tokens_nums =
      torch::tensor(kv_cache_tokens_nums, torch::kInt);
  input_params.new_cache_slots = torch::tensor(new_cache_slots, torch::kInt);
  input_params.paged_kv_indptr = torch::tensor(paged_kv_indptr, torch::kInt);
  input_params.paged_kv_indices = torch::tensor(paged_kv_indices, torch::kInt);
  input_params.paged_kv_last_page_len =
      torch::tensor(paged_kv_last_page_len, torch::kInt);
  input_params.block_tables = create_2d_tensor(block_tables, torch::kInt);

  return input;
}

}  // namespace runtime
}  // namespace xllm
