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
#include "framework/request/mm_data.h"
#include "npu_dp_ep_padding.h"
#include "util/tensor_helper.h"

namespace xllm {
struct CacheBlockInfo {
  int32_t device_block_id = 0;
  int32_t host_block_id = 0;
  uint8_t* hash_key = nullptr;

  CacheBlockInfo() {}

  CacheBlockInfo(int32_t device_block_id, int32_t host_block_id) {
    this->device_block_id = device_block_id;
    this->host_block_id = host_block_id;
  }

  CacheBlockInfo(int32_t device_block_id,
                 int32_t host_block_id,
                 const uint8_t* hash_key) {
    this->device_block_id = device_block_id;
    this->host_block_id = host_block_id;
    this->hash_key = const_cast<uint8_t*>(hash_key);
  }
};

struct ModelInputParams {
  ModelInputParams to(const torch::Device& device) const {
    ModelInputParams params;
    params.empty_kv_cache = empty_kv_cache;
    params.global_empty_kv_cache = global_empty_kv_cache;
    params.num_sequences = num_sequences;
    params.kv_max_seq_len = kv_max_seq_len;
    params.q_max_seq_len = q_max_seq_len;

    params.kv_seq_lens = safe_to(kv_seq_lens, device, true);
    params.q_seq_lens = safe_to(q_seq_lens, device, true);

    params.new_cache_slots = safe_to(new_cache_slots, device, true);
    params.block_tables = safe_to(block_tables, device, true);
    params.kv_seq_lens_vec = kv_seq_lens_vec;
    params.q_seq_lens_vec = q_seq_lens_vec;
    params.decode_seq_range = decode_seq_range;

    params.input_embedding = safe_to(input_embedding, device);

    params.deep_stacks = deep_stacks;
    params.visual_pos_masks = visual_pos_masks;

    params.mm_data = MMData::to(mm_data, device);
    params.dp_global_token_nums = dp_global_token_nums;
    params.prefill_seq_len = prefill_seq_len;
    params.embedding_ids = std::move(embedding_ids);
    params.extra_token_ids = std::move(extra_token_ids);
    params.dp_ep_padding_data = dp_ep_padding_data;
#if defined(USE_NPU)
    params.layer_synchronizer = layer_synchronizer;
#endif
    params.expert_load_data = expert_load_data;

    params.async_copy_out_blocks = std::move(async_copy_out_blocks);
    params.copy_out_blocks = std::move(copy_out_blocks);
    params.copy_in_blocks = std::move(copy_in_blocks);
    params.swap_blocks = std::move(swap_blocks);

    params.src_block_indices = safe_to(src_block_indices, device, true);
    params.dst_block_indices = safe_to(dst_block_indices, device, true);
    params.cum_sum = safe_to(cum_sum, device, true);

    // params for continuous kvcache
    params.new_cache_slot_offsets = safe_to(new_cache_slot_offsets, device);
    params.kv_cache_start_offsets = safe_to(kv_cache_start_offsets, device);

    // Copy graph_buffer to device
    params.graph_buffer = safe_to(graph_buffer, device, true);

    // params for flashinfer
    params.paged_kv_indptr = safe_to(paged_kv_indptr, device);
    params.paged_kv_indices = safe_to(paged_kv_indices, device);
    params.paged_kv_last_page_len = safe_to(paged_kv_last_page_len, device);

    return params;
  }

  void print() const {
    LOG(INFO) << "ModelInputParams: empty_kv_cache is " << empty_kv_cache
              << " , global_empty_kv_cache is " << global_empty_kv_cache
              << " , num_sequences is " << num_sequences
              << " , kv_max_seq_len is " << kv_max_seq_len
              << " , q_max_seq_len is " << q_max_seq_len
              << " , prefill_seq_len is " << prefill_seq_len;
    LOG(INFO) << "ModelInputParams: kv_seq_lens_vec is " << kv_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: q_seq_lens_vec is " << q_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: decode_seq_range is " << decode_seq_range;
    print_tensor(kv_seq_lens, "ModelInputParams: kv_seq_lens", 4);
    print_tensor(q_seq_lens, "ModelInputParams: q_seq_lens", 4);
    print_tensor(new_cache_slots, "ModelInputParams: new_cache_slots", 4);
    print_tensor(block_tables, "ModelInputParams: block_tables", 4);
    LOG(INFO) << "ModelInputParams: dp_global_token_nums is "
              << dp_global_token_nums;
  }
  // whether the kv-cache is empty for all sequences.
  bool empty_kv_cache = true;

  // total number of sequences in the batch
  int32_t num_sequences = 0;

  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  std::vector<int> kv_seq_lens_vec;
  std::vector<int> q_seq_lens_vec;
  // Range of decode sequence indices in the batch [start, end].
  // Decode sequences are identified by q_seq_lens == 1,
  // prefill sequences by  q_seq_lens > 1 .
  // Used to determine whether to use prefill_node_ or
  // decode_node_ in NPU layers
  // Values: {-1, -1} if no decode requests (all prefill),
  //         {0, batch_size-1} if all decode requests,
  //         {start_idx, end_idx} if mixed prefill/decode requests
  std::pair<int, int> decode_seq_range;
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
  MMData mm_data;

  // deep_stack for Qwen3-VL
  mutable std::vector<torch::Tensor> deep_stacks;
  // visual pos mask for Qwen3-VL
  mutable torch::Tensor visual_pos_masks;

  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  // whether the kv-cache is empty for all sequences,mainly used for dp case
  bool global_empty_kv_cache = true;

  // num of prefill sequence in chunked prefill case
  uint32_t prefill_seq_len = 0;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;

  // copy in / copy out
  std::vector<CacheBlockInfo> async_copy_out_blocks;
  std::vector<CacheBlockInfo> copy_out_blocks;
  std::vector<CacheBlockInfo> copy_in_blocks;
  std::vector<CacheBlockInfo> swap_blocks;

  // block copy kernel
  torch::Tensor src_block_indices;
  torch::Tensor dst_block_indices;
  torch::Tensor cum_sum;

#if defined(USE_NPU)
  std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer = nullptr;
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
  // Graph execution buffer for temporary tensor storage
  // Used by ACL Graph Executor to avoid repeated memory allocation
  torch::Tensor graph_buffer;

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
};

}  // namespace xllm
