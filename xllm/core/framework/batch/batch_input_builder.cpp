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

#include "batch_input_builder.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/batch/mposition.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/params_utils.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

void split_copy_out_blocks(RawForwardInput& raw_forward_input,
                           std::unordered_set<int32_t>& write_block_ids) {
  std::vector<CacheBlockInfo> async_copy_out_blocks;
  std::vector<CacheBlockInfo> sync_copy_out_blocks;
  for (CacheBlockInfo content : raw_forward_input.copy_out_blocks) {
    if (write_block_ids.find(content.device_block_id) !=
        write_block_ids.end()) {
      sync_copy_out_blocks.push_back(content);
    } else {
      async_copy_out_blocks.push_back(content);
    }
  }
  raw_forward_input.copy_out_blocks = std::move(sync_copy_out_blocks);
  raw_forward_input.async_copy_out_blocks = std::move(async_copy_out_blocks);
}

BatchInputBuilder::BatchInputBuilder(
    const std::vector<Sequence*>& sequences,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
    const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
    const ModelArgs* args)
    : sequences_(sequences),
      allowed_max_tokens_(allowed_max_tokens),
      input_embeddings_vec_(input_embeddings_vec),
      mm_data_vec_(mm_data_vec),
      args_(args),
      num_sequences_(static_cast<int32_t>(sequences.size())),
      copy_in_cache_block_infos_(copy_in_cache_block_infos),
      copy_out_cache_block_infos_(copy_out_cache_block_infos) {
  // Reserve space for better performance
  state_.flatten_tokens_vec.reserve(1000);
  state_.flatten_positions_vec.reserve(1000);
  state_.mrope_positions_vec.reserve(sequences.size());
  state_.block_tables_vec.reserve(sequences.size());
  if (args_ != nullptr) {
    use_mrope_ = (args_->rope_scaling_rope_type() == "mrope");
  }
  write_block_ids_.clear();
}

ForwardInput BatchInputBuilder::build_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  process_sequences();
  padding_decode_batch_size(num_decoding_tokens, min_decoding_batch_size);

  return state_to_forward_input();
}

RawForwardInput BatchInputBuilder::build_raw_forward_input() {
  process_sequences();
  return state_to_raw_forward_input();
}

void BatchInputBuilder::process_sequences() {
  for (int32_t i = 0; i < num_sequences_; ++i) {
    process_single_sequence(i);
  }
}

void BatchInputBuilder::process_single_sequence(int32_t seq_index) {
  auto* sequence = sequences_[seq_index];
  const auto token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();

  // Validate and calculate sequence lengths
  CHECK(allowed_max_tokens_[seq_index] > 0);
  const uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
  const uint32_t seq_len = q_seq_len + n_kv_cache_tokens;

  // Validation
  CHECK_GE(sequence->kv_state().current_max_tokens_capacity(), seq_len);
  CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                         << "n_tokens: " << n_tokens
                         << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                         << ", current_max_tokens_capacity: "
                         << sequence->kv_state().current_max_tokens_capacity()
                         << ", allowed_max_tokens: "
                         << allowed_max_tokens_[seq_index];

  // Update state
  state_.empty_kv_cache = state_.empty_kv_cache && (n_kv_cache_tokens == 0);
  state_.max_seq_len = std::max(state_.max_seq_len, seq_len);
  state_.q_max_seq_len = std::max(state_.q_max_seq_len, q_seq_len);
#if defined(USE_NPU)
  state_.seq_lens.push_back(seq_len);
  state_.q_seq_lens.push_back(q_seq_len);
#elif defined(USE_MLU)
  state_.seq_lens.push_back(state_.seq_lens.back() + seq_len);
  state_.q_seq_lens.push_back(state_.q_seq_lens.back() + q_seq_len);
#endif
  // Process tokens and positions
  extract_tokens_and_positions(sequence, n_kv_cache_tokens, seq_len);

  // Setup KV cache
  setup_kv_cache_info(sequence, n_kv_cache_tokens, seq_len, q_seq_len);

  // Track prefill sequences
  if (sequence->is_prefill_stage()) {
    state_.prefill_seq_len++;
  }
  state_.embedding_ids.push_back(sequence->get_embedding_id());
}

void BatchInputBuilder::extract_tokens_and_positions(Sequence* sequence,
                                                     uint32_t n_kv_cache_tokens,
                                                     uint32_t seq_len) {
  const auto& token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();

  // Prepare adjusted token counts for sampling
  std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    // skip prompt tokens except the last one
    if (j + 1 < n_tokens) continue;
    ++adjusted_token_to_count_map[token_ids[j]];
  }

  // Handle MRope positions
  if (use_mrope_) {
    const auto& args = *args_;
    MPositionHelper helper(*sequence, args);
    state_.mrope_positions_vec.push_back(helper.get_positions());
  }

  // Process each token
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    state_.flatten_tokens_vec.push_back(token_ids[j]);

    if (!use_mrope_) {
      state_.flatten_positions_vec.push_back(static_cast<int32_t>(j));
    }

    // Handle sampling for last tokens
    if (j + 1 < n_tokens) continue;

    handle_sampling_parameters(
        sequence, j, seq_len, adjusted_token_to_count_map);
  }
}

void BatchInputBuilder::handle_sampling_parameters(
    Sequence* sequence,
    uint32_t token_position,
    uint32_t seq_len,
    std::unordered_map<int32_t, int32_t>& adjusted_token_to_count_map) {
  // const auto token_ids = sequence->token_ids();
  const auto token_id = sequence->tokens()[token_position];

  // Adjust token count
  --adjusted_token_to_count_map[token_id];

  // Select token for sampling
  state_.selected_token_idxes.push_back(state_.flatten_tokens_vec.size() - 1);
  state_.sampling_params.push_back(sequence->sampling_param());

  // Process unique tokens
  const auto& seq_token_counts = sequence->token_to_count_map();
  auto& ids = state_.unique_token_ids_vec.emplace_back();
  auto& counts = state_.unique_token_counts_vec.emplace_back();

  ids.reserve(seq_token_counts.size());
  counts.reserve(seq_token_counts.size());

  for (const auto& [token_id, count] : seq_token_counts) {
    const auto it = adjusted_token_to_count_map.find(token_id);
    const auto adjust_count =
        (it != adjusted_token_to_count_map.end()) ? it->second : 0;

    if (count > adjust_count) {
      ids.push_back(token_id);
      counts.push_back(count - adjust_count);
    }
  }

  state_.unique_token_lens_vec.push_back(static_cast<int32_t>(ids.size()));

  // Mark sample token if it's the last token
  // TODO add test
  // in chunked prefill condition, if allowed_max_token = 128, n_tokens=1000,
  // n_kv_cache_tokens=256, q_seq_len = 128, seq_len=384
  if (token_position == seq_len - 1) {
    state_.sample_idxes.push_back(
        static_cast<int32_t>(state_.selected_token_idxes.size() - 1));
  }
}

void BatchInputBuilder::setup_kv_cache_info(Sequence* sequence,
                                            uint32_t n_kv_cache_tokens,
                                            uint32_t seq_len,
                                            uint32_t q_seq_len) {
  // update kv cache tokens num
  sequence->kv_state().incr_kv_cache_tokens_num(/*size=*/q_seq_len);

  const auto blocks = sequence->kv_state().kv_blocks();
  const auto slot_ids =
      sequence->kv_state().kv_cache_slots(n_kv_cache_tokens, seq_len);
  state_.new_token_slot_ids.insert(
      state_.new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

  std::vector<int32_t> block_ids;
  std::vector<uint64_t> u_block_ids;
  block_ids.reserve(blocks.size());
  int32_t block_size = 0;
  for (const auto& block : blocks) {
    block_size = block.size();
    block_ids.push_back(block.id());
    u_block_ids.emplace_back(block.id());
  }

  int32_t kv_cache_block_idx = n_kv_cache_tokens / block_size;
  for (auto iter = block_ids.begin() + kv_cache_block_idx;
       iter != block_ids.end();
       ++iter) {
    write_block_ids_.insert(*iter);
  }

  auto& transfer_kv_info = sequence->kv_state().transfer_kv_info();
  if (transfer_kv_info.has_value()) {
    state_.transfer_kv_infos.emplace_back(transfer_kv_info.value());
    state_.transfer_kv_infos.back().local_blocks_ids = std::move(u_block_ids);
  }

  state_.block_tables_vec.emplace_back(std::move(block_ids));
}

void BatchInputBuilder::setup_continuous_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len) {
  // update kv cache tokens num
  sequence->kv_state().incr_kv_cache_tokens_num(/*size=*/q_seq_len);

  int32_t seq_id = sequence->seq_id();

  int64_t kv_cache_start_offset = seq_id * FLAGS_buffer_size_per_seq;
  std::vector<int64_t> cache_slot_offsets;
  cache_slot_offsets.reserve(seq_len - n_kv_cache_tokens);
  for (int32_t i = n_kv_cache_tokens; i < seq_len; ++i) {
    cache_slot_offsets.push_back(kv_cache_start_offset +
                                 i * FLAGS_cache_size_per_token);
  }
  state_.new_cache_slot_offsets.insert(state_.new_cache_slot_offsets.end(),
                                       cache_slot_offsets.begin(),
                                       cache_slot_offsets.end());
  state_.kv_cache_start_offsets.push_back(kv_cache_start_offset);
}

void BatchInputBuilder::padding_decode_batch_size(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  if (num_sequences_ < min_decoding_batch_size) {
    const uint32_t n_tokens = state_.flatten_tokens_vec.size();
    // kv_cache is not empty in decoding phase
    const bool in_decoding_phase = !state_.empty_kv_cache;
    const bool same_num_decoding_tokens =
        state_.q_max_seq_len == num_decoding_tokens &&
        n_tokens == num_sequences_ * num_decoding_tokens;
    if (in_decoding_phase && same_num_decoding_tokens) {
      // add padding tokens to the batch
      for (int32_t i = num_sequences_; i < min_decoding_batch_size; ++i) {
        for (int32_t k = 0; k < num_decoding_tokens; ++k) {
          state_.flatten_tokens_vec.push_back(0);
          if (!use_mrope_) {
            state_.flatten_positions_vec.push_back(0);
          } else {
            state_.mrope_positions_vec.push_back(
                torch::zeros({3, 1}, torch::kInt));
          }
          state_.new_token_slot_ids.push_back(0);
        }
#if defined(USE_NPU)
        state_.seq_lens.push_back(num_decoding_tokens);
        state_.q_seq_lens.push_back(num_decoding_tokens);
#elif defined(USE_MLU)
        state_.seq_lens.push_back(state_.seq_lens.back() + num_decoding_tokens);
        state_.q_seq_lens.push_back(state_.q_seq_lens.back() +
                                    num_decoding_tokens);
#endif
        state_.block_tables_vec.emplace_back();
      }
    }
  }
}

ForwardInput BatchInputBuilder::state_to_forward_input() {
  if (state_.flatten_tokens_vec.empty()) {
    return {};
  }

  ForwardInput forward_input;

  // Create tensors
  forward_input.token_ids =
      torch::tensor(state_.flatten_tokens_vec, torch::kInt);

  if (!use_mrope_) {
    forward_input.positions =
        torch::tensor(state_.flatten_positions_vec, torch::kInt);
  } else {
    forward_input.positions = torch::cat(state_.mrope_positions_vec, 1);
  }

  auto& input_params = forward_input.input_params;
  input_params.empty_kv_cache = state_.empty_kv_cache;
  input_params.num_sequences = state_.block_tables_vec.size();
  input_params.kv_max_seq_len = state_.max_seq_len;
  input_params.q_max_seq_len = state_.q_max_seq_len;
  input_params.kv_seq_lens = torch::tensor(state_.seq_lens, torch::kInt);
  input_params.q_seq_lens = torch::tensor(state_.q_seq_lens, torch::kInt);
  input_params.kv_seq_lens_vec = std::move(state_.seq_lens);
  input_params.q_seq_lens_vec = std::move(state_.q_seq_lens);
  input_params.new_cache_slots =
      torch::tensor(state_.new_token_slot_ids, torch::kInt);
  input_params.prefill_indices =
      util::find_ones_indices(input_params.q_seq_lens_vec);

  // Setup multimodal data
  input_params.mm_data = MMData::batch(mm_data_vec_);

  // Setup block tables
  util::pad_2d_vector(state_.block_tables_vec, /*pad_value=*/0);
  input_params.block_tables =
      create_2d_tensor(state_.block_tables_vec, torch::kInt);

  if (input_embeddings_vec_.size() != 0) {
    input_params.input_embedding = torch::cat(input_embeddings_vec_);
  }

  if (FLAGS_enable_continuous_kvcache) {
    input_params.new_cache_slots =
        torch::tensor(state_.new_cache_slot_offsets, torch::kInt64);
    input_params.kv_cache_start_offsets =
        torch::tensor(state_.kv_cache_start_offsets, torch::kInt64);
  }

  CHECK_EQ(state_.sampling_params.size(), state_.selected_token_idxes.size());
  // Setup sampling parameters
  if (!state_.selected_token_idxes.empty()) {
    util::pad_2d_vector<int64_t>(state_.unique_token_ids_vec, /*pad_value=*/0);
    util::pad_2d_vector(state_.unique_token_counts_vec, /*pad_value=*/0);

    forward_input.sampling_params.init(state_.sampling_params,
                                       state_.selected_token_idxes,
                                       state_.sample_idxes,
                                       state_.unique_token_ids_vec,
                                       state_.unique_token_counts_vec,
                                       state_.unique_token_lens_vec);
  }

  return forward_input;
}

RawForwardInput BatchInputBuilder::state_to_raw_forward_input() {
  if (state_.flatten_tokens_vec.empty()) {
    return {};
  }
  RawForwardInput raw_forward_input;
  raw_forward_input.flatten_tokens_vec = std::move(state_.flatten_tokens_vec);
  raw_forward_input.flatten_positions_vec =
      std::move(state_.flatten_positions_vec);
  raw_forward_input.sampling_params = std::move(state_.sampling_params);
  raw_forward_input.selected_token_idxes =
      std::move(state_.selected_token_idxes);
  raw_forward_input.sample_idxes = std::move(state_.sample_idxes);
  raw_forward_input.unique_token_ids_vec =
      std::move(state_.unique_token_ids_vec);
  raw_forward_input.unique_token_counts_vec =
      std::move(state_.unique_token_counts_vec);
  raw_forward_input.unique_token_lens_vec =
      std::move(state_.unique_token_lens_vec);
  raw_forward_input.empty_kv_cache = state_.empty_kv_cache;
  // raw_forward_input.global_empty_kv_cache = ;
  raw_forward_input.max_seq_len = state_.max_seq_len;
  raw_forward_input.q_max_seq_len = state_.q_max_seq_len;
  raw_forward_input.seq_lens = std::move(state_.seq_lens);
  raw_forward_input.q_seq_lens = std::move(state_.q_seq_lens);
  raw_forward_input.new_token_slot_ids = std::move(state_.new_token_slot_ids);
  raw_forward_input.block_tables_vec = std::move(state_.block_tables_vec);
  raw_forward_input.num_sequences = num_sequences_;
  // raw_forward_input.dp_global_token_nums = ;
  raw_forward_input.transfer_kv_infos = std::move(state_.transfer_kv_infos);
  raw_forward_input.prefill_seq_len = state_.prefill_seq_len;

  raw_forward_input.embedding_ids = std::move(state_.embedding_ids);

  if (FLAGS_enable_continuous_kvcache) {
    raw_forward_input.new_cache_slot_offsets =
        std::move(state_.new_cache_slot_offsets);
    raw_forward_input.kv_cache_start_offsets =
        std::move(state_.kv_cache_start_offsets);
  }

  if (mm_data_vec_.size() != 0) {
    MMData mm_data = MMData::batch(mm_data_vec_);
    const auto& res = mm_data.get<torch::Tensor>("embedding");
    if (res) {
      torch::Tensor embeddings = res.value();
      for (int64_t output_idx = 0; output_idx < embeddings.size(0);
           ++output_idx) {
        torch::Tensor embedding = embeddings[output_idx].to(torch::kFloat32);
        Slice<float> embedding_slice = {embedding.data_ptr<float>(),
                                        embedding.size(0)};
        raw_forward_input.embeddings.push_back(embedding_slice);
      }
    }
  }

  if (copy_out_cache_block_infos_ != nullptr &&
      copy_out_cache_block_infos_->size() > 0) {
    raw_forward_input.copy_out_blocks.insert(
        raw_forward_input.copy_out_blocks.end(),
        copy_out_cache_block_infos_->begin(),
        copy_out_cache_block_infos_->end());
  }
  if (copy_in_cache_block_infos_ != nullptr &&
      copy_in_cache_block_infos_->size() > 0) {
    raw_forward_input.copy_in_blocks.insert(
        raw_forward_input.copy_in_blocks.end(),
        copy_in_cache_block_infos_->begin(),
        copy_in_cache_block_infos_->end());
  }

  split_copy_out_blocks(raw_forward_input, write_block_ids_);

  return raw_forward_input;
}

}  // namespace xllm
