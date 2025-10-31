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

// batch_input_builder.h
#pragma once

#include <torch/torch.h>

#include <limits>
#include <vector>

#include "framework/request/mm_data.h"
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

struct ModelArgs;

class BatchInputBuilder {
 public:
  explicit BatchInputBuilder(
      const std::vector<Sequence*>& sequences,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      // for global kv cache copy block from host to device
      const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
      // for global kv cache copy block from device to host
      const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
      // for beam-search
      std::vector<CacheBlockInfo>* swap_cache_block_infos,
      const ModelArgs* args,
      ThreadPool* thread_pool = nullptr);

  ForwardInput build_forward_input(uint32_t num_decoding_tokens,
                                   uint32_t min_decoding_batch_size);

  RawForwardInput build_raw_forward_input(uint32_t start_idx, uint32_t end_idx);

 private:
  // Core building methods
  void process_sequences(uint32_t start_idx, uint32_t end_idx);
  void process_sequences_multithreaded(uint32_t start_idx, uint32_t end_idx);
  void padding_decode_batch_size(uint32_t num_decoding_tokens,
                                 uint32_t min_decoding_batch_size);
  ForwardInput state_to_forward_input();
  RawForwardInput state_to_raw_forward_input();

  void process_swap_block_infos(RawForwardInput& raw_forward_input);

  // State management
  struct BuilderState {
    // Token and position data
    std::vector<int32_t> flatten_tokens_vec;
    std::vector<int32_t> flatten_positions_vec;
    std::vector<torch::Tensor> mrope_positions_vec;

    // Sampling data
    std::vector<const RequestSamplingParam*> sampling_params;
    std::vector<int32_t> selected_token_idxes;
    std::vector<int32_t> sample_idxes;

    // Unique token tracking
    std::vector<std::vector<int64_t>> unique_token_ids_vec;
    std::vector<std::vector<int32_t>> unique_token_counts_vec;
    std::vector<int32_t> unique_token_lens_vec;

    // Sequence metadata
    bool empty_kv_cache = true;
    uint32_t max_seq_len = 0;
    uint32_t q_max_seq_len = 0;
#if defined(USE_NPU)
    std::vector<int32_t> seq_lens;
    std::vector<int32_t> q_seq_lens;
#elif defined(USE_MLU)
    std::vector<int32_t> seq_lens = {0};    // cu_seq_lens
    std::vector<int32_t> q_seq_lens = {0};  // q_cu_seq_len
#endif

    // Cache and block data
    std::vector<int32_t> new_token_slot_ids;
    std::vector<std::vector<int32_t>> block_tables_vec;

    // beam search kernel input
    std::vector<float> acc_logprob_vec;

    // Additional data
    std::vector<int32_t> embedding_ids;
    std::vector<int32_t> extra_token_ids;
    uint32_t prefill_seq_len = 0;
    std::vector<TransferKVInfo> transfer_kv_infos;

    // for continuous kvcache
    std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
    std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]
  };

  // Helper methods for sequence processing
  void process_single_sequence(
      int32_t seq_index,
      BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr);
  void extract_tokens_and_positions(Sequence* sequence,
                                    uint32_t n_kv_cache_tokens,
                                    uint32_t seq_len,
                                    BuilderState* state_ptr = nullptr);
  void handle_sampling_parameters(
      Sequence* sequence,
      uint32_t token_position,
      uint32_t seq_len,
      std::unordered_map<int32_t, int32_t>& adjusted_counts,
      BuilderState* state_ptr = nullptr);
  void setup_kv_cache_info(
      Sequence* sequence,
      uint32_t n_kv_cache_tokens,
      uint32_t seq_len,
      uint32_t q_seq_len,
      BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr);
  void setup_continuous_kv_cache_info(Sequence* sequence,
                                      uint32_t n_kv_cache_tokens,
                                      uint32_t seq_len,
                                      uint32_t q_seq_len,
                                      BuilderState* state_ptr = nullptr);

 protected:
  // Input data
  const std::vector<Sequence*>& sequences_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  const std::vector<MMData>& mm_data_vec_;
  const ModelArgs* args_;

  // Builder state
  BuilderState state_;

  // Configuration
  bool use_mrope_ = false;
  int32_t num_sequences_ = 0;

  // copy in and out cache contents
  std::unordered_set<int32_t> write_block_ids_;
  const std::vector<CacheBlockInfo>* copy_in_cache_block_infos_ = nullptr;
  const std::vector<CacheBlockInfo>* copy_out_cache_block_infos_ = nullptr;
  std::vector<CacheBlockInfo>* swap_cache_block_infos_ = nullptr;

  // thread pool for multithreaded processing, not owned
  ThreadPool* thread_pool_ = nullptr;
};

}  // namespace xllm
