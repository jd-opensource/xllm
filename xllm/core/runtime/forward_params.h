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

#include <algorithm>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "framework/request/mm_batch_data.h"
#include "framework/request/mm_data.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "runtime/dit_forward_params.h"

namespace xllm {

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,     // LLM
    VLM,     // VLM
    DIT,     // DIT
    ELM,     // Embedding LM
    EVLM,    // Embedding VLM
    REC,     // Rec
    MMEVLM,  // Encoder Embedding VLM
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
    } else if (str == "MMEVLM") {
      value_ = MMEVLM;
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
    } else if (this->value_ == MMEVLM) {
      return "MMEVLM";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Step-level decode metadata for Rec multi-round (device loop).
struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  // Planned decode kv cache shape: [batch_size * beam_width, n_kv_heads,
  // step_rounds, head_dim]
  std::vector<int64_t> full_kv_shape;
  // Flattened decode positions for each sequence.
  std::vector<int32_t> decode_positions_vec;
};

// Inputs for forward execution
struct ForwardInput {
  ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    ForwardInput inputs;
    inputs.token_ids = safe_to(token_ids, device, true);
    inputs.positions = safe_to(positions, device, true);
    // Convert positions to int64 on CUDA/ILU/MUSA to avoid repeated per-layer
    // type conversions in rope kernels.
    const auto dev = Device::type_str();
    if ((dev == "cuda" || dev == "ilu" || dev == "musa") &&
        inputs.positions.defined() &&
        inputs.positions.scalar_type() != torch::kInt64) {
      inputs.positions = inputs.positions.to(torch::kInt64);
    }
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.eplb_info = eplb_info;
    inputs.acc_logprob = safe_to(acc_logprob, device, true);
    inputs.step_decode = step_decode;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.device_input_buffer = device_input_buffer;
    return inputs;
  }

  void print() const {
    LOG(INFO) << "  token_ids: " << token_ids << std::endl;
    LOG(INFO) << "  positions: " << positions << std::endl;
    input_params.print();
    LOG(INFO) << " params.selected_token_idxes "
              << sampling_params.selected_token_idxes;
    LOG(INFO) << " params.sample_idxes " << sampling_params.sample_idxes;
    LOG(INFO) << " params.do_sample " << sampling_params.do_sample;
  }

  const StepDecodeMeta* step_meta() const {
    return step_decode ? &(*step_decode) : nullptr;
  }

  bool has_step_meta() const { return step_decode.has_value(); }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;
  // beam search kernel input
  torch::Tensor acc_logprob;

  // step-level decode metadata
  std::optional<StepDecodeMeta> step_decode;
  // If true, skip sampler forward and only keep logits.
  bool skip_sampling_for_logits_only = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;

  // A tensor used to store all device-side input data, with other input tensors
  // constructed based on the address and offset of this tensor.
  torch::Tensor device_input_buffer;
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
  torch::Tensor beam_sequence_group;

  // dit output data
  DiTForwardOutput dit_forward_output;
};

// Model input with raw data, which will be
// serielize to pb type before pass to remote worker.
struct RawForwardInput {
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  std::vector<std::vector<int32_t>> m_positions_vec;
  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  std::vector<int32_t> unique_token_lens_vec;
  BatchForwardType batch_forward_type;
  uint32_t max_seq_len;
  uint32_t q_max_seq_len;
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t num_sequences;
  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  std::vector<int32_t> dp_is_decode;
  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;
  std::vector<std::vector<float>> embeddings;
  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;
  // embedding ids of each sequence
  std::vector<int> embedding_ids;
  // linear state ids of each sequence
  std::vector<int> linear_state_ids;
  // request ids of each sequence
  std::vector<std::string> request_ids;
  // swap
  std::vector<BlockTransferInfo> swap_blocks;
  uint64_t batch_id;
  // block copy kernel
  std::vector<int32_t> src_block_indices;
  std::vector<int32_t> dst_block_indices;
  std::vector<int32_t> cum_sum;
  // for continuous kvcache
  std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
  std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]
  // beam search kernel input
  std::vector<float> acc_logprob_vec;
  // for flashinfer
  std::vector<int32_t> paged_kv_indptr;         //[n_seq + 1]
  std::vector<int32_t> paged_kv_indices;        //[num_used_pages]
  std::vector<int32_t> paged_kv_last_page_len;  //[n_seq]
  // multimodal data
  MMBatchData mm_data;

  // dit input data
  DiTForwardInput dit_forward_input;

  RawForwardInput cp_partition(int32_t cp_rank, int32_t cp_size) const {
    RawForwardInput outputs = *this;
    if (cp_size <= 1 || flatten_tokens_vec.empty() ||
        !batch_forward_type.is_prefill()) {
      return outputs;
    }

    CHECK_GT(cp_size, 0);
    CHECK_GE(cp_rank, 0);
    CHECK_LT(cp_rank, cp_size);
    CHECK_GT(num_sequences, 0);

    const int32_t num_chunks = cp_size * 2;
    const int64_t token_num = static_cast<int64_t>(flatten_tokens_vec.size());

    auto to_seq_lens =
        [&](const std::vector<int32_t>& lens) -> std::vector<int32_t> {
      if (lens.empty()) {
        return std::vector<int32_t>(num_sequences, 0);
      }
      const bool is_cumsum =
          lens.size() == static_cast<size_t>(num_sequences + 1) &&
          lens.front() == 0;
      std::vector<int32_t> seq_lens;
      seq_lens.reserve(num_sequences);
      if (is_cumsum) {
        for (int32_t i = 0; i < num_sequences; ++i) {
          seq_lens.push_back(std::max(0, lens[i + 1] - lens[i]));
        }
      } else {
        CHECK_GE(lens.size(), static_cast<size_t>(num_sequences));
        for (int32_t i = 0; i < num_sequences; ++i) {
          seq_lens.push_back(std::max(0, lens[i]));
        }
      }
      return seq_lens;
    };

    const std::vector<int32_t> input_lens =
        !q_seq_lens.empty() ? to_seq_lens(q_seq_lens) : to_seq_lens(seq_lens);

    std::vector<int32_t> cp_q_lens;
    cp_q_lens.reserve(num_sequences);
    std::vector<int64_t> gather_indices;
    gather_indices.reserve(token_num);
    int32_t cp_global_max_seq_len = 0;

    std::vector<int64_t> old_seq_offsets;
    old_seq_offsets.reserve(num_sequences + 1);
    old_seq_offsets.push_back(0);
    std::vector<int64_t> new_seq_offsets;
    new_seq_offsets.reserve(num_sequences + 1);
    new_seq_offsets.push_back(0);

    for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      const int32_t input_len = std::max(0, input_lens[seq_idx]);
      const int64_t seq_start = old_seq_offsets.back();
      const int64_t chunk_len =
          (input_len + num_chunks - 1) / static_cast<int64_t>(num_chunks);

      auto range_len = [&](int64_t local_start, int64_t local_end) -> int64_t {
        local_start = std::max<int64_t>(0, local_start);
        local_end = std::max<int64_t>(0, local_end);
        local_start = std::min<int64_t>(local_start, input_len);
        local_end = std::min<int64_t>(local_end, input_len);
        return std::max<int64_t>(0, local_end - local_start);
      };

      int64_t local_len = 0;
      auto append_range = [&](int64_t local_start, int64_t local_end) {
        const int64_t valid_len = range_len(local_start, local_end);
        if (valid_len <= 0) {
          return;
        }
        const int64_t start =
            std::max<int64_t>(0, std::min<int64_t>(local_start, input_len));
        for (int64_t i = 0; i < valid_len; ++i) {
          gather_indices.push_back(seq_start + start + i);
        }
        local_len += valid_len;
      };

      append_range(chunk_len * cp_rank, chunk_len * (cp_rank + 1));
      append_range(chunk_len * (num_chunks - 1 - cp_rank),
                   chunk_len * (num_chunks - cp_rank));

      cp_q_lens.push_back(static_cast<int32_t>(local_len));
      old_seq_offsets.push_back(seq_start + input_len);
      new_seq_offsets.push_back(new_seq_offsets.back() + local_len);

      int64_t seq_cp_max = 0;
      for (int32_t rank = 0; rank < cp_size; ++rank) {
        const int64_t former_len =
            range_len(chunk_len * rank, chunk_len * (rank + 1));
        const int64_t latter_len =
            range_len(chunk_len * (num_chunks - 1 - rank),
                      chunk_len * (num_chunks - rank));
        seq_cp_max = std::max(seq_cp_max, former_len + latter_len);
      }
      cp_global_max_seq_len =
          std::max(cp_global_max_seq_len, static_cast<int32_t>(seq_cp_max));
    }
    CHECK_EQ(old_seq_offsets.back(), token_num);

    auto gather_token_level_vector_i32 = [&](const std::vector<int32_t>& src) {
      if (src.size() != static_cast<size_t>(token_num)) {
        return src;
      }
      std::vector<int32_t> dst;
      dst.reserve(gather_indices.size());
      for (int64_t idx : gather_indices) {
        dst.push_back(src[static_cast<size_t>(idx)]);
      }
      return dst;
    };

    outputs.flatten_tokens_vec =
        gather_token_level_vector_i32(flatten_tokens_vec);
    if (!flatten_positions_vec.empty()) {
      outputs.flatten_positions_vec =
          gather_token_level_vector_i32(flatten_positions_vec);
    }

    auto build_seq_lens = [&](const std::vector<int32_t>& original,
                              const std::vector<int32_t>& lengths) {
      const bool is_cumsum =
          original.size() == static_cast<size_t>(num_sequences + 1) &&
          !original.empty() && original.front() == 0;
      std::vector<int32_t> result;
      if (is_cumsum) {
        result.reserve(num_sequences + 1);
        result.push_back(0);
        for (const int32_t len : lengths) {
          result.push_back(result.back() + len);
        }
      } else {
        result.assign(lengths.begin(), lengths.end());
      }
      return result;
    };

    outputs.q_seq_lens = build_seq_lens(q_seq_lens, cp_q_lens);
    outputs.seq_lens = build_seq_lens(seq_lens, cp_q_lens);
    outputs.q_cu_seq_lens.resize(cp_q_lens.size());
    std::partial_sum(
        cp_q_lens.begin(), cp_q_lens.end(), outputs.q_cu_seq_lens.begin());

    outputs.q_max_seq_len = cp_global_max_seq_len;
    outputs.max_seq_len = cp_global_max_seq_len;

    if (!selected_token_idxes.empty()) {
      const int64_t selected_num =
          static_cast<int64_t>(selected_token_idxes.size());
      std::vector<int64_t> remapped_idxes;
      remapped_idxes.reserve(selected_num);

      const int64_t num_chunks_i64 = static_cast<int64_t>(cp_size) * 2;
      std::vector<int64_t> seq_context_lens(num_sequences, 0);
      std::vector<int64_t> selected_seq_idx(selected_num, 0);

      for (int64_t i = 0; i < selected_num; ++i) {
        const int64_t old_idx = selected_token_idxes[i];
        auto upper = std::upper_bound(
            old_seq_offsets.begin(), old_seq_offsets.end(), old_idx);
        int64_t seq_idx =
            static_cast<int64_t>(upper - old_seq_offsets.begin()) - 1;
        seq_idx = std::max<int64_t>(
            0,
            std::min<int64_t>(seq_idx,
                              static_cast<int64_t>(num_sequences) - 1));
        selected_seq_idx[i] = seq_idx;

        const int64_t seq_start = old_seq_offsets[seq_idx];
        const int64_t seq_end = old_seq_offsets[seq_idx + 1];
        const int64_t seq_len = std::max<int64_t>(0, seq_end - seq_start);
        const int64_t context_len = std::max<int64_t>(
            1, std::min<int64_t>(old_idx - seq_start + 1, seq_len));
        seq_context_lens[seq_idx] =
            std::max(seq_context_lens[seq_idx], context_len);
      }

      std::vector<int64_t> chunk_lens(num_sequences, 1);
      std::vector<int64_t> seq_prefix_per_rank(num_sequences, 0);
      int64_t token_num_per_rank = 0;

      for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        int64_t chunk_len =
            (seq_context_lens[seq_idx] + num_chunks_i64 - 1) / num_chunks_i64;
        chunk_len = std::max<int64_t>(1, chunk_len);
        chunk_lens[seq_idx] = chunk_len;
        seq_prefix_per_rank[seq_idx] = token_num_per_rank;
        token_num_per_rank += (chunk_len * num_chunks_i64) / cp_size;
      }

      remapped_idxes.clear();
      for (int64_t i = 0; i < selected_num; ++i) {
        const int64_t old_idx = selected_token_idxes[i];
        const int64_t seq_idx = selected_seq_idx[i];
        const int64_t seq_start = old_seq_offsets[seq_idx];
        const int64_t seq_context_len = seq_context_lens[seq_idx];
        const int64_t chunk_len = chunk_lens[seq_idx];

        int64_t token_pos = old_idx - seq_start;
        token_pos = std::max<int64_t>(
            0, std::min<int64_t>(token_pos, seq_context_len - 1));
        const int64_t chunk_id = token_pos / chunk_len;
        const int64_t offset = token_pos % chunk_len;
        const int64_t rank_id =
            chunk_id >= cp_size
                ? static_cast<int64_t>(2 * cp_size) - chunk_id - 1
                : chunk_id;
        const int64_t remap_idx = token_num_per_rank * rank_id +
                                  seq_prefix_per_rank[seq_idx] +
                                  (chunk_id / cp_size) * chunk_len + offset;
        remapped_idxes.push_back(remap_idx);
      }

      outputs.selected_token_idxes.clear();
      outputs.selected_token_idxes.reserve(remapped_idxes.size());
      for (int64_t idx : remapped_idxes) {
        outputs.selected_token_idxes.push_back(static_cast<int32_t>(idx));
      }
    }

    return outputs;
  }
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

  // batch-level beam output for Rec multi-round mode
  std::vector<int32_t> beam_sequence_group;  // flattened 2D
  // multimodal embedding output
  std::vector<torch::Tensor> mm_embeddings;
  // dit output data
  DiTForwardOutput dit_forward_output;
};

struct BatchedForwardInputs {
  std::vector<ForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
  // beam search kernel input
  torch::Tensor acc_logprob;
};

}  // namespace xllm
