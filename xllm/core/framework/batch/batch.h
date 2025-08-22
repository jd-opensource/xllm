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
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"

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

  void set_copy_in_cache_contents(
      std::vector<CacheBlockInfo>* copy_in_cache_contents) {
    copy_in_cache_contents_ = copy_in_cache_contents;
  }

  void set_copy_out_cache_contents(
      std::vector<CacheBlockInfo>* copy_out_cache_contents) {
    copy_out_cache_contents_ = copy_out_cache_contents;
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
  RawForwardInput prepare_forward_input();

  // process output
  void process_sample_output(const SampleOutput& sample_output,
                             bool enable_schedule_overlap);

  void process_sample_output(const RawForwardOutput& raw_output,
                             bool enable_schedule_overlap);

  // process the accepted output embedding
  void process_embedding_output(const torch::Tensor& embedding);

  // split the whole batch into several micro batches
  std::vector<Batch> split(const size_t num_micro_batches);

  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return allowed_max_tokens_;
  }

 private:
  bool update_sequence_state(Sequence* seq, bool enable_schedule_overlap);

  void append_token_for_sequence(Sequence* seq,
                                 const Token& token,
                                 int token_idx,
                                 bool enable_schedule_overlap);

  std::vector<Sequence*> sequences_;
  std::vector<CacheBlockInfo>* copy_in_cache_contents_ = nullptr;
  std::vector<CacheBlockInfo>* copy_out_cache_contents_ = nullptr;

  // max number of tokens to process for each sequence
  // default to max value
  std::vector<uint32_t> allowed_max_tokens_;

  std::vector<torch::Tensor> input_embeddings_vec_;

  // mm_data in the batch
  std::vector<MMData> mm_data_vec_;
};

}  // namespace xllm
