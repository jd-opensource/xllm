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

#include <nlohmann/json.hpp>
#include <optional>

#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampling_params.h"

namespace xllm {

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,   // LLM
    VLM,   // VLM
    DIT,   // DIT
    ELM,   // Embedding LM
    EVLM,  // Embedding VLM
    REC,   // Rec
  };

  constexpr WorkerType(Value v) : value_(v) {}
  WorkerType(const std::string& str) {
    if (str == "LLM") {
      value_ = LLM;
    } else if (str == "VLM") {
      value_ = VLM;
    } else if (str == "DIT") {
      value_ = DIT;
    } else if (str == "ELM") {
      value_ = ELM;
    } else if (str == "EVLM") {
      value_ = EVLM;
    } else if (str == "REC") {
      value_ = REC;
    } else {
      value_ = INVALID;
    }
  }

  WorkerType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(WorkerType rhs) const { return value_ == rhs.value_; }
  bool operator!=(WorkerType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else if (this->value_ == ELM) {
      return "ELM";
    } else if (this->value_ == EVLM) {
      return "EVLM";
    } else if (this->value_ == REC) {
      return "REC";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Inputs for forward execution
struct ForwardInput {
  ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    ForwardInput inputs;
    inputs.token_ids = safe_to(token_ids, device, true);
    inputs.positions = safe_to(positions, device, true);
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.eplb_info = eplb_info;
    inputs.acc_logprob = safe_to(acc_logprob, device, true);
    return inputs;
  }
  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;
  // beam search kernel input
  torch::Tensor acc_logprob;
};

// output after forward execution
struct ForwardOutput {
  // sample parameters for speculative decoding
  torch::Tensor do_sample;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  torch::Tensor logits;
  torch::Tensor embedding;

  // for eplb, collect the tokens load of experts on each worker.
  torch::Tensor expert_load_data;
  // for eplb, indicates that the specified layer on the worker
  // has completed the asynchronous loading of new weight.
  int32_t prepared_layer_id;

  BeamSearchOutput beam_search_output;
};

// Model input with raw data, which will be
// serielize to pb type before pass to remote worker.
struct RawForwardInput {
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  std::vector<int32_t> unique_token_lens_vec;
  bool empty_kv_cache = true;
  bool global_empty_kv_cache = true;
  uint32_t max_seq_len;
  uint32_t q_max_seq_len;
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t num_sequences;
  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;
  std::vector<std::vector<float>> embeddings;
  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;
  // num of prefill sequence in chunked prefill case
  uint32_t prefill_seq_len;
  // embedding ids of each sequence
  std::vector<int> embedding_ids;
  // copy in / copy out
  std::vector<CacheBlockInfo> async_copy_out_blocks;
  std::vector<CacheBlockInfo> copy_out_blocks;
  std::vector<CacheBlockInfo> copy_in_blocks;
  std::vector<CacheBlockInfo> swap_blocks;
  // block copy kernel
  std::vector<int32_t> src_block_indices;
  std::vector<int32_t> dst_block_indices;
  std::vector<int32_t> cum_sum;
  // for continuous kvcache
  std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
  std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]
  // beam search kernel input
  std::vector<float> acc_logprob_vec;
};

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
};

struct RawForwardOutput {
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<int64_t> expert_load_data;
  int32_t prepared_layer_id;
  // beam search kernel output
  std::vector<int32_t> src_seq_idxes;
  std::vector<int32_t> out_tokens;
  std::vector<float> out_logprobs;
};

struct BatchedForwardInputs {
  std::vector<ForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
  // beam search kernel input
  torch::Tensor acc_logprob;
};

}  // namespace xllm
