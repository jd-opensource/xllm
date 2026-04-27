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

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model_input_params.h"

namespace xllm {
namespace model_input {

struct LLMModelInputParams {
  bool enable_mla = false;
  BatchForwardType batch_forward_type;
  int32_t num_sequences = 0;
  int32_t kv_max_seq_len = 0;
  int32_t q_max_seq_len = 0;
  uint64_t batch_id = 0;

  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  std::vector<int> kv_seq_lens_vec;
  std::vector<int> q_seq_lens_vec;

  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor new_cache_slot_offsets;
  torch::Tensor kv_cache_start_offsets;
  torch::Tensor input_embedding;

  std::vector<int32_t> dp_global_token_nums;
  std::vector<int32_t> dp_is_decode;
  std::vector<int32_t> embedding_ids;
  std::vector<std::string> request_ids;
  std::vector<int32_t> extra_token_ids;
  std::vector<BlockTransferInfo> swap_blocks;

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
  CpEpPaddingData cp_ep_padding_data;
  CpPrefillInputs cp_prefill_inputs;

  torch::Tensor expert_load_data;
  torch::Tensor expert_array;
  torch::Tensor kv_cache_tokens_nums;
  std::vector<int32_t> kv_cache_tokens_nums_host;
  torch::Tensor history_compressed_kv;
  torch::Tensor history_k_rope;
  torch::Tensor ring_cur_seqlen;
  std::vector<int32_t> ring_cur_seqlen_host;
  torch::Tensor ring_cache_seqlen;
  std::vector<int32_t> ring_cache_seqlen_host;
  torch::Tensor graph_attn_mask;
  torch::Tensor graph_tiling_data;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  bool enable_cuda_graph = false;

  static LLMModelInputParams from_legacy(const xllm::ModelInputParams& src);
  void apply_to_legacy(xllm::ModelInputParams* dst) const;
};

struct VLMModelInputParams {
  MMBatchData mm_data;
  std::vector<torch::Tensor> deep_stacks;
  torch::Tensor visual_pos_masks;

  static VLMModelInputParams from_legacy(const xllm::ModelInputParams& src);
  void apply_to_legacy(xllm::ModelInputParams* dst) const;
};

struct DitModelInputParams {
  DiTForwardInput dit_forward_input;

  static DitModelInputParams from_legacy(const xllm::ModelInputParams& src);
  void apply_to_legacy(xllm::ModelInputParams* dst) const;
};

struct RecModelInputParams {
  xllm::RecModelInputParams rec_params;

  static RecModelInputParams from_legacy(const xllm::ModelInputParams& src);
  void apply_to_legacy(xllm::ModelInputParams* dst) const;
};

struct ModelInputParamBundle {
  std::shared_ptr<LLMModelInputParams> llm;
  std::shared_ptr<VLMModelInputParams> vlm;
  std::shared_ptr<DitModelInputParams> dit;
  std::shared_ptr<RecModelInputParams> rec;

  static ModelInputParamBundle from_legacy(const xllm::ModelInputParams& src);
  void apply_to_legacy(xllm::ModelInputParams* dst) const;
};

}  // namespace model_input
}  // namespace xllm
