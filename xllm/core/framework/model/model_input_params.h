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

#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#include "framework/batch/batch_forward_type.h"
#include "framework/request/mm_batch_data.h"
#include "npu_dp_ep_padding.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"

namespace xllm {

enum class TransferType : uint8_t {
  G2H = 0,  // global memory(KVCache store) to host memory(DRAM)
  H2D = 1,  // host memory(DRAM) to device memory(HBM)
  D2G = 2,  // host memory(DRAM) to global memory(KVCache store)
  G2D = 3   // global memory(KVCache store) to device memory(HBM)
};

struct BlockTransferInfo {
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  uint8_t hash_key[MURMUR_HASH3_VALUE_LEN];
  TransferType transfer_type;

  BlockTransferInfo(int32_t src_block_id, int32_t dst_block_id) {
    this->src_block_id = src_block_id;
    this->dst_block_id = dst_block_id;
  }

  BlockTransferInfo(int32_t src_id,
                    int32_t dst_id,
                    const uint8_t* key,
                    TransferType type)
      : src_block_id(src_id), dst_block_id(dst_id), transfer_type(type) {
    memcpy(hash_key, key, MURMUR_HASH3_VALUE_LEN);
  }

  BlockTransferInfo(const BlockTransferInfo& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, MURMUR_HASH3_VALUE_LEN);
  }

  BlockTransferInfo(BlockTransferInfo&& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, MURMUR_HASH3_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
  }

  BlockTransferInfo& operator=(const BlockTransferInfo& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, MURMUR_HASH3_VALUE_LEN);
    return *this;
  }

  BlockTransferInfo& operator=(BlockTransferInfo&& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, MURMUR_HASH3_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
    return *this;
  }

  std::string to_string() const {
    std::string rt = ", has_key:";
    for (int i = 0; i < 16; i++) {
      rt += std::to_string(int64_t(hash_key[i])) + " ";
    }
    return std::to_string(src_block_id) + "->" + std::to_string(dst_block_id) +
           ", " + std::to_string(uint32_t(transfer_type)) + rt;
  }
};

struct ModelInputParams {
  ModelInputParams to(const torch::Device& device) const {
    ModelInputParams params;
    params.is_prefill = is_prefill;
    params.empty_kv_cache = empty_kv_cache;
    params.global_empty_kv_cache = global_empty_kv_cache;
    params.batch_forward_type = batch_forward_type;
    params.num_sequences = num_sequences;
    params.kv_max_seq_len = kv_max_seq_len;
    params.q_max_seq_len = q_max_seq_len;

    params.kv_seq_lens = safe_to(kv_seq_lens, device, true);
    params.q_seq_lens = safe_to(q_seq_lens, device, true);
    params.q_cu_seq_lens = safe_to(q_cu_seq_lens, device, true);

    params.new_cache_slots = safe_to(new_cache_slots, device, true);
    params.block_tables = safe_to(block_tables, device, true);
    params.kv_seq_lens_vec = kv_seq_lens_vec;
    params.q_seq_lens_vec = q_seq_lens_vec;
    params.decode_kv_seq_lens = safe_to(decode_kv_seq_lens, device, true);
    params.decode_q_seq_lens = safe_to(decode_q_seq_lens, device, true);
    params.decode_kv_seq_lens_vec = decode_kv_seq_lens_vec;
    params.decode_q_seq_lens_vec = decode_q_seq_lens_vec;

    params.input_embedding = safe_to(input_embedding, device);

    params.deep_stacks = deep_stacks;
    params.visual_pos_masks = visual_pos_masks;

    params.mm_data = mm_data.to(device);
    params.dp_global_token_nums = dp_global_token_nums;
    params.embedding_ids = std::move(embedding_ids);
    params.extra_token_ids = std::move(extra_token_ids);
    params.dp_ep_padding_data = dp_ep_padding_data;
#if defined(USE_NPU)
    params.layer_synchronizer = layer_synchronizer;
#endif
    params.expert_load_data = expert_load_data;

    params.swap_blocks = std::move(swap_blocks);

    params.src_block_indices = safe_to(src_block_indices, device, true);
    params.dst_block_indices = safe_to(dst_block_indices, device, true);
    params.cum_sum = safe_to(cum_sum, device, true);

    // params for continuous kvcache
    params.new_cache_slot_offsets = safe_to(new_cache_slot_offsets, device);
    params.kv_cache_start_offsets = safe_to(kv_cache_start_offsets, device);

    // shared kv caches per layer (optional)
    params.shared_k_caches.clear();
    params.shared_v_caches.clear();
    for (const auto& t : shared_k_caches) {
      params.shared_k_caches.push_back(safe_to(t, device));
    }
    for (const auto& t : shared_v_caches) {
      params.shared_v_caches.push_back(safe_to(t, device));
    }
    params.beam_width_tensor = safe_to(beam_width_tensor, device);
    params.current_round_tensor = safe_to(current_round_tensor, device);
    params.current_round_tensor_list.clear();
    for (const auto& t : current_round_tensor_list) {
      params.current_round_tensor_list.push_back(safe_to(t, device));
    }
    params.decode_positions_tensor_list.clear();
    for (const auto& t : decode_positions_tensor_list) {
      params.decode_positions_tensor_list.push_back(safe_to(t, device));
    }

    params.beam_width = beam_width;
    params.current_round = current_round;
    params.total_round = total_round;
    // Copy graph_buffer to device
    // params.graph_buffer = safe_to(graph_buffer, device, true);
    params.graph_buffer.attn_mask =
        safe_to(graph_buffer.attn_mask, device, true);
    params.graph_buffer.tiling_data =
        safe_to(graph_buffer.tiling_data, device, true);

    // Copy rec graph buffer tensor to device
    params.graph_buffer_rec = safe_to(graph_buffer_rec, device, true);

    // params for flashinfer
    params.paged_kv_indptr = safe_to(paged_kv_indptr, device);
    params.paged_kv_indices = safe_to(paged_kv_indices, device);
    params.paged_kv_last_page_len = safe_to(paged_kv_last_page_len, device);

    params.batch_id = batch_id;

    return params;
  }

  void print() const {
    LOG(INFO) << "ModelInputParams: empty_kv_cache is " << empty_kv_cache
              << " , global_empty_kv_cache is " << global_empty_kv_cache
              << " , num_sequences is " << num_sequences
              << " , kv_max_seq_len is " << kv_max_seq_len
              << " , q_max_seq_len is " << q_max_seq_len;
    LOG(INFO) << "ModelInputParams: kv_seq_lens_vec is " << kv_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: q_seq_lens_vec is " << q_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << batch_forward_type.to_string();
    print_tensor(kv_seq_lens, "ModelInputParams: kv_seq_lens", 4);
    print_tensor(q_seq_lens, "ModelInputParams: q_seq_lens", 4);
    print_tensor(q_cu_seq_lens, "ModelInputParams: q_cu_seq_lens", 4);
    print_tensor(decode_kv_seq_lens, "ModelInputParams: decode_kv_seq_lens", 4);
    print_tensor(decode_q_seq_lens, "ModelInputParams: decode_q_seq_lens", 4);
    print_tensor(new_cache_slots, "ModelInputParams: new_cache_slots", 4);
    print_tensor(block_tables, "ModelInputParams: block_tables", 4);
    LOG(INFO) << "ModelInputParams: dp_global_token_nums is "
              << dp_global_token_nums;
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
#if defined(USE_NPU)
    CHECK(seq_idx < q_seq_lens_vec.size()) << "seq_idx out of range";
    return q_seq_lens_vec[seq_idx];
#else
    CHECK(seq_idx < q_seq_lens_vec.size() - 1) << "seq_idx out of range";
    return q_seq_lens_vec[seq_idx + 1] - q_seq_lens_vec[seq_idx];
#endif
  }

  bool synchronize_layer(uint32_t layer_idx) const {
#if defined(USE_NPU)
    if (layer_wise_load_synchronizer != nullptr &&
        layer_idx % layers_per_bacth_copy == 0) {
      if (!layer_wise_load_synchronizer->synchronize_layer(
              layer_idx / layers_per_bacth_copy)) {
        return false;
      }
    }
#endif
    return true;
  }

  // whether the kv-cache is empty for all sequences.
  bool empty_kv_cache = true;

  // whether this pass is prefill stage
  bool is_prefill = true;
  BatchForwardType batch_forward_type;

  // total number of sequences in the batch
  int32_t num_sequences = 0;

  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  std::vector<int> kv_seq_lens_vec;
  std::vector<int> q_seq_lens_vec;
  torch::Tensor decode_q_seq_lens;
  torch::Tensor decode_kv_seq_lens;
  std::vector<int> decode_kv_seq_lens_vec;
  std::vector<int> decode_q_seq_lens_vec;

  // max length for qkv.
  int32_t kv_max_seq_len = 0;
  int32_t q_max_seq_len = 0;

  // IntTensor: [n_tokens]
  torch::Tensor new_cache_slots;

  // IntTensor: [n_seq, max_n_blocks]
  torch::Tensor block_tables;

  // input embedding
  mutable torch::Tensor input_embedding;

  // multimodal
  MMBatchData mm_data;

  // deep_stack for Qwen3-VL
  mutable std::vector<torch::Tensor> deep_stacks;
  // visual pos mask for Qwen3-VL
  mutable torch::Tensor visual_pos_masks;

  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  // whether the kv-cache is empty for all sequences,mainly used for dp case
  bool global_empty_kv_cache = true;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;

  // swap
  std::vector<BlockTransferInfo> swap_blocks;

  // block copy kernel
  torch::Tensor src_block_indices;
  torch::Tensor dst_block_indices;
  torch::Tensor cum_sum;

#if defined(USE_NPU)
  std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer = nullptr;
  uint32_t layers_per_bacth_copy = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<NPULayerSynchronizerImpl> layer_wise_load_synchronizer =
      nullptr;
#endif

  DpEpPaddingData dp_ep_padding_data;
  torch::Tensor expert_load_data;

  // new slot offsets for continuous kvcache
  // used to store kv-cache to right position
  // IntTensor: [n_tokens]
  torch::Tensor new_cache_slot_offsets;

  // kvcache offset of sequence in the xtensor for all layers
  // IntTensor: [n_seq]
  torch::Tensor kv_cache_start_offsets;

  // the indptr of the paged kv-cache
  // used in flashinfer
  // IntTensor: [n_seq + 1]
  torch::Tensor paged_kv_indptr;

  // the page indices of the paged kv cache
  // used in flashinfer
  torch::Tensor paged_kv_indices;

  // the number of entries in the last page of each request in
  // the paged kv cache
  // used in flashinfer
  // IntTensor: [n_seq]
  torch::Tensor paged_kv_last_page_len;

  uint64_t batch_id;

  struct GraphBuffer {
    torch::Tensor attn_mask;
    torch::Tensor tiling_data;
  };
  GraphBuffer graph_buffer;

  torch::Tensor graph_buffer_rec;

  // shared kv caches provided by engine for step-level decode, per layer
  std::vector<torch::Tensor> shared_k_caches;
  std::vector<torch::Tensor> shared_v_caches;
  torch::Tensor beam_width_tensor;
  torch::Tensor current_round_tensor;
  std::vector<torch::Tensor> current_round_tensor_list;
  std::vector<torch::Tensor> decode_positions_tensor_list;
  // beam width for step-level decode
  int32_t beam_width = 1;
  // current round for step-level decode
  int32_t current_round = 0;
  int32_t total_round = 0;
};

}  // namespace xllm
