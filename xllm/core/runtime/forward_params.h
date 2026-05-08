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
#include <cstring>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/global_flags.h"
#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "framework/request/mm_batch_data.h"
#include "framework/request/mm_data.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "runtime/dit_forward_params.h"

namespace xllm {

struct ForwardInput;

namespace detail {

constexpr uint64_t kForwardInputBufferAlignment = 16;

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

inline bool supports_contiguous_forward_input_buffer(
    const torch::Device& device) {
#if defined(USE_CUDA)
  return device.type() == torch::kCUDA;
#elif defined(USE_MLU)
  return device.type() == torch::kPrivateUse1;
#elif defined(USE_NPU)
  (void)device;
  return true;
#else
  (void)device;
  return false;
#endif
}

bool try_to_device_from_input_host_buffer(const ForwardInput& input,
                                          const torch::Device& device,
                                          torch::ScalarType dtype,
                                          ForwardInput& output);

struct ForwardInputBufferEntry {
  torch::Tensor host_tensor;
  torch::Tensor* target = nullptr;
  uint64_t offset = 0;
  uint64_t aligned_bytes = 0;
};

struct ForwardInputBufferPlan {
  std::vector<ForwardInputBufferEntry> entries;

  bool add(const torch::Tensor& tensor, torch::Tensor* target) {
    if (!tensor.defined()) {
      return true;
    }
    if (!tensor.device().is_cpu()) {
      return false;
    }
    entries.push_back({tensor.contiguous(), target, 0, 0});
    return true;
  }

  uint64_t prepare_layout() {
    uint64_t total = 0;
    for (auto& entry : entries) {
      total = align_up(total, kForwardInputBufferAlignment);
      entry.offset = total;
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      entry.aligned_bytes = align_up(bytes, kForwardInputBufferAlignment);
      total += entry.aligned_bytes;
    }
    return total;
  }

  torch::Tensor build_host_buffer(uint64_t total_bytes) const {
    auto buffer = torch::empty({static_cast<int64_t>(total_bytes)},
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
    auto* base = static_cast<char*>(buffer.data_ptr());
    for (const auto& entry : entries) {
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      if (bytes == 0) {
        continue;
      }
      std::memcpy(base + entry.offset, entry.host_tensor.data_ptr(), bytes);
      if (entry.aligned_bytes > bytes) {
        std::memset(base + entry.offset + bytes,
                    0,
                    static_cast<size_t>(entry.aligned_bytes - bytes));
      }
    }
    return buffer;
  }

  void bind_device_views(const torch::Tensor& device_buffer,
                         const torch::Device& device) const {
    const char* base = static_cast<const char*>(device_buffer.data_ptr());
    for (const auto& entry : entries) {
      if (entry.target == nullptr || !entry.host_tensor.defined()) {
        continue;
      }
      const void* ptr = base + entry.offset;
#if defined(USE_CUDA)
      if (device.type() == torch::kCUDA) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_MLU)
      if (device.type() == torch::kPrivateUse1) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_NPU)
      *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                           entry.host_tensor.scalar_type(),
                                           ptr);
#else
      (void)device;
#endif
    }
  }
};

inline bool add_sampling_to_plan(const SamplingParameters& source,
                                 SamplingParameters& target,
                                 ForwardInputBufferPlan& plan) {
  return plan.add(source.selected_token_idxes, &target.selected_token_idxes) &&
         plan.add(source.frequency_penalties, &target.frequency_penalties) &&
         plan.add(source.presence_penalties, &target.presence_penalties) &&
         plan.add(source.repetition_penalties, &target.repetition_penalties) &&
         plan.add(source.temperatures, &target.temperatures) &&
         plan.add(source.top_p, &target.top_p) &&
         plan.add(source.top_k, &target.top_k) &&
         plan.add(source.unique_token_ids, &target.unique_token_ids) &&
         plan.add(source.unique_token_counts, &target.unique_token_counts) &&
         plan.add(source.unique_token_ids_lens,
                  &target.unique_token_ids_lens) &&
         plan.add(source.sample_idxes, &target.sample_idxes) &&
         plan.add(source.do_sample, &target.do_sample) &&
         plan.add(source.acc_logprob, &target.acc_logprob);
}

inline torch::Tensor normalize_positions_for_device(
    const torch::Tensor& positions) {
  const auto dev = Device::type_str();
  if ((dev == "cuda" || dev == "ilu" || dev == "musa") && positions.defined() &&
      positions.scalar_type() != torch::kInt64) {
    return positions.to(torch::kInt64);
  }
  return positions;
}

inline bool has_contiguous_input_buffer_exclusions(
    const ModelInputParams& params) {
  return params.multimodal.mm_data.valid() || params.has_onerec_params() ||
         params.has_llmrec_params() || params.dit_forward_input.valid() ||
         params.multimodal.visual_pos_masks.defined() ||
         !params.multimodal.deep_stacks.empty();
}

inline void clear_contiguous_input_buffer_tensor_targets(
    ModelInputParams& params) {
  params.embedding.input_embedding = torch::Tensor();
  params.embedding.linear_state_indices = torch::Tensor();
  params.block_copy.src_block_indices = torch::Tensor();
  params.block_copy.dst_block_indices = torch::Tensor();
  params.block_copy.cum_sum = torch::Tensor();
  params.graph.attn_mask = torch::Tensor();
  params.graph.tiling_data = torch::Tensor();
}

inline bool add_attention_to_plan(const AttentionInput& source,
                                  AttentionInput& target,
                                  ForwardInputBufferPlan& plan) {
  return plan.add(source.device.q_seq_lens, &target.device.q_seq_lens) &&
         plan.add(source.device.kv_seq_lens, &target.device.kv_seq_lens) &&
         plan.add(source.device.q_cu_seq_lens, &target.device.q_cu_seq_lens) &&
         plan.add(source.device.new_cache_slots,
                  &target.device.new_cache_slots) &&
         plan.add(source.device.block_tables, &target.device.block_tables) &&
         plan.add(source.device.paged_kv_indptr,
                  &target.device.paged_kv_indptr) &&
         plan.add(source.device.paged_kv_indices,
                  &target.device.paged_kv_indices) &&
         plan.add(source.device.paged_kv_last_page_len,
                  &target.device.paged_kv_last_page_len) &&
         plan.add(source.device.new_cache_slot_offsets,
                  &target.device.new_cache_slot_offsets) &&
         plan.add(source.device.kv_cache_start_offsets,
                  &target.device.kv_cache_start_offsets) &&
         plan.add(source.device.kv_cache_tokens_nums,
                  &target.device.kv_cache_tokens_nums) &&
         plan.add(source.device.history_compressed_kv,
                  &target.device.history_compressed_kv) &&
         plan.add(source.device.history_k_rope,
                  &target.device.history_k_rope) &&
         plan.add(source.device.ring_cur_seqlen,
                  &target.device.ring_cur_seqlen) &&
         plan.add(source.device.ring_cache_seqlen,
                  &target.device.ring_cache_seqlen);
}

inline bool add_model_tensors_to_plan(const ModelInputParams& source,
                                      ModelInputParams& target,
                                      ForwardInputBufferPlan& plan) {
  return plan.add(source.embedding.input_embedding,
                  &target.embedding.input_embedding) &&
         plan.add(source.embedding.linear_state_indices,
                  &target.embedding.linear_state_indices) &&
         plan.add(source.block_copy.src_block_indices,
                  &target.block_copy.src_block_indices) &&
         plan.add(source.block_copy.dst_block_indices,
                  &target.block_copy.dst_block_indices) &&
         plan.add(source.block_copy.cum_sum, &target.block_copy.cum_sum) &&
         plan.add(source.graph.attn_mask, &target.graph.attn_mask) &&
         plan.add(source.graph.tiling_data, &target.graph.tiling_data);
}

}  // namespace detail

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
    if (device_tensors_ready) {
      return *this;
    }

    if (input_host_buffer_has_layout && FLAGS_use_contiguous_input_buffer &&
        detail::supports_contiguous_forward_input_buffer(device)) {
      ForwardInput buffer_inputs;
      if (detail::try_to_device_from_input_host_buffer(
              *this, device, dtype, buffer_inputs)) {
        return buffer_inputs;
      }
    }

    if (FLAGS_use_contiguous_input_buffer &&
        detail::supports_contiguous_forward_input_buffer(device)) {
      ForwardInput contiguous_inputs;
      if (to_contiguous_input_buffer(device, contiguous_inputs)) {
        return contiguous_inputs;
      }
    }

    ForwardInput inputs;
    set_host_views(inputs);
    const torch::Tensor& source_token_ids =
        inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
    const torch::Tensor& source_positions =
        inputs.positions_host.defined() ? inputs.positions_host : positions;
    inputs.token_ids = safe_to(source_token_ids, device, true);
    inputs.positions = detail::normalize_positions_for_device(
        safe_to(source_positions, device, true));
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    copy_metadata_to(inputs);
    inputs.input_host_buffer = input_host_buffer;
    inputs.device_input_buffer = device_input_buffer;
    inputs.input_host_buffer_has_layout = input_host_buffer_has_layout;
    inputs.device_tensors_ready = true;
    return inputs;
  }

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  ForwardInput& inputs) const {
    copy_metadata_to(inputs);
    set_host_views(inputs);

    const ModelInputParams& source_params = input_params;
    if (missing_required_host_views(inputs) ||
        detail::has_contiguous_input_buffer_exclusions(source_params)) {
      return false;
    }

    inputs.input_params = source_params;
    detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);

    inputs.sampling_params = sampling_params;
    inputs.decoder_sampling_params = decoder_sampling_params;

    torch::Tensor positions_for_device =
        detail::normalize_positions_for_device(inputs.positions_host);

    detail::ForwardInputBufferPlan plan;
    if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
        !plan.add(positions_for_device, &inputs.positions)) {
      return false;
    }

    if (!detail::add_attention_to_plan(
            source_params.attention, inputs.input_params.attention, plan) ||
        !detail::add_model_tensors_to_plan(
            source_params, inputs.input_params, plan)) {
      return false;
    }

    if (!detail::add_sampling_to_plan(
            sampling_params, inputs.sampling_params, plan) ||
        !detail::add_sampling_to_plan(
            decoder_sampling_params, inputs.decoder_sampling_params, plan)) {
      return false;
    }

    const uint64_t total_bytes = plan.prepare_layout();
    if (total_bytes > 0) {
      inputs.input_host_buffer = plan.build_host_buffer(total_bytes);
      inputs.device_input_buffer =
          safe_to(inputs.input_host_buffer,
                  torch::TensorOptions().dtype(torch::kUInt8).device(device),
                  true);
      plan.bind_device_views(inputs.device_input_buffer, device);
    }

    inputs.device_tensors_ready = true;
    inputs.input_host_buffer_has_layout = false;
    return true;
  }

  void copy_metadata_to(ForwardInput& inputs) const {
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.step_decode = step_decode;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
  }

  void set_host_views(ForwardInput& inputs) const {
    inputs.token_ids_host =
        token_ids_host.defined() ? token_ids_host : cpu_view(token_ids);
    inputs.positions_host =
        positions_host.defined() ? positions_host : cpu_view(positions);
  }

  bool missing_required_host_views(const ForwardInput& inputs) const {
    return (token_ids.defined() && !inputs.token_ids_host.defined()) ||
           (positions.defined() && !inputs.positions_host.defined());
  }

  const torch::Tensor& host_token_ids() const {
    return token_ids_host.defined() ? token_ids_host : token_ids;
  }

  const torch::Tensor& host_positions() const {
    return positions_host.defined() ? positions_host : positions;
  }

  static torch::Tensor cpu_view(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.device().is_cpu()) {
      return tensor;
    }
    return torch::Tensor();
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
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;

  // step-level decode metadata
  std::optional<StepDecodeMeta> step_decode;
  // If true, skip sampler forward and only keep logits.
  bool skip_sampling_for_logits_only = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;

  // A tensor used to store all device-side input data, with other input tensors
  // constructed based on the address and offset of this tensor.
  torch::Tensor input_host_buffer;
  torch::Tensor device_input_buffer;
  bool input_host_buffer_has_layout = false;

  // True when token_ids, positions, model input tensors and sampling tensors
  // already point to the device-side views for execution. Worker prepare can
  // then skip rebuilding/H2D in ForwardInput::to().
  bool device_tensors_ready = false;
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
};

}  // namespace xllm
