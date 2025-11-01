/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <limits>
#include <vector>

#include "framework/request/mm_data.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

struct ModelArgs;

class Batch {
 public:
  Batch() = default;

  Batch(Sequence* sequence);
  Batch(const std::vector<Sequence*>& sequences);

  void add(Sequence* sequence,
           uint32_t allowed_max_token = std::numeric_limits<uint32_t>::max());

  void add(const std::vector<Sequence*>& sequences);

  void add(SequencesGroup* sequence_group) {
    sequence_groups_.push_back(sequence_group);
  }

  void set_copy_in_cache_block_infos(
      std::vector<CacheBlockInfo>* copy_in_cache_block_infos) {
    copy_in_cache_block_infos_ = copy_in_cache_block_infos;
  }

  void set_copy_out_cache_block_infos(
      std::vector<CacheBlockInfo>* copy_out_cache_block_infos) {
    copy_out_cache_block_infos_ = copy_out_cache_block_infos;
  }

  void set_swap_cache_block_infos(
      std::vector<CacheBlockInfo>* swap_cache_block_infos) {
    swap_cache_block_infos_ = swap_cache_block_infos;
  }

  // get the number of sequences in the batch
  size_t size() const { return sequences_.size(); }
  bool empty() const { return sequences_.empty(); }

  Sequence* operator[](size_t i) { return sequences_[i]; }

  // prepare forward inputs
  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_bach_size,
                                     const ModelArgs& args);

  // Convert Batch to pb type, which will be pass to remote worker.
  RawForwardInput prepare_forward_input(uint32_t start_idx,
                                        uint32_t end_idx,
                                        ThreadPool* thread_pool = nullptr);

  // process output
  //
  // replace_fake_token:
  // In the scenario where enable_schedule_overlap is true,
  // the forward is divided into two stages.
  // The first stage populates the sequence with a fake token,
  // and the second stage replaces the previous fake token with a real token.
  // The boolean parameter `replace_fake_token` indicates
  // whether the current stage is the second stage.
  void process_sample_output(const SampleOutput& sample_output,
                             bool replace_fake_token);

  void process_sample_output(const RawForwardOutput& raw_output,
                             bool replace_fake_token);

  // process output for beam search kernel
  void process_beam_search_output(const RawForwardOutput& raw_output,
                                  bool replace_fake_token);

  // process the accepted output embedding
  void process_embedding_output(const torch::Tensor& embedding);

  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return allowed_max_tokens_;
  }

  void set_batch_prefill_status(const bool all_seqs_in_prefill) {
    all_seqs_in_prefill_ = all_seqs_in_prefill;
  }

  bool get_batch_prefill_status() const { return all_seqs_in_prefill_; }

  void finish();

  // prepare forward inputs for Rec model
  ForwardInput prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                         uint32_t min_decoding_batch_size,
                                         const ModelArgs& args,
                                         ThreadPool* thread_pool = nullptr);

 private:
  bool update_sequence_state(Sequence* seq, bool replace_fake_token);

  void append_token_for_sequence(Sequence* seq,
                                 const Token& token,
                                 int token_idx,
                                 bool replace_fake_token);

  void process_beam_search();

  std::vector<Sequence*> sequences_;
  std::vector<SequencesGroup*> sequence_groups_;
  std::vector<CacheBlockInfo>* copy_in_cache_block_infos_ = nullptr;
  std::vector<CacheBlockInfo>* copy_out_cache_block_infos_ = nullptr;
  std::vector<CacheBlockInfo>* swap_cache_block_infos_ = nullptr;

  // max number of tokens to process for each sequence
  // default to max value
  std::vector<uint32_t> allowed_max_tokens_;

  std::vector<torch::Tensor> input_embeddings_vec_;

  // mm_data in the batch
  std::vector<MMData> mm_data_vec_;

  // all sequences in this batch are in prefill stage
  bool all_seqs_in_prefill_ = false;
};

}  // namespace xllm
