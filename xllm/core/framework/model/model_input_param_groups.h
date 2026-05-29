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
#include <optional>
#include <string>
#include <vector>

#include "model_input_params.h"

namespace xllm {
namespace model_input {

struct LLMModelInputParams {
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
};

struct VLMModelInputParams {
  MMBatchData mm_data;
  std::vector<torch::Tensor> deep_stacks;
  torch::Tensor visual_pos_masks;
};

struct DitModelInputParams {
  DiTForwardInput dit_forward_input;
};

struct RecModelInputParams {
  xllm::RecModelInputParams rec_params;
};

struct ModelInputParamBundle {
  std::optional<LLMModelInputParams> llm;
  std::optional<VLMModelInputParams> vlm;
  std::optional<DitModelInputParams> dit;
  std::optional<RecModelInputParams> rec;
};

LLMModelInputParams make_llm_model_input_params_from_legacy(
    const xllm::ModelInputParams& src);
LLMModelInputParams make_llm_model_input_params_from_legacy(
    xllm::ModelInputParams&& src);
void apply_llm_model_input_params_to_legacy(const LLMModelInputParams& src,
                                            xllm::ModelInputParams* dst);

VLMModelInputParams make_vlm_model_input_params_from_legacy(
    const xllm::ModelInputParams& src);
VLMModelInputParams make_vlm_model_input_params_from_legacy(
    xllm::ModelInputParams&& src);
void apply_vlm_model_input_params_to_legacy(const VLMModelInputParams& src,
                                            xllm::ModelInputParams* dst);

DitModelInputParams make_dit_model_input_params_from_legacy(
    const xllm::ModelInputParams& src);
DitModelInputParams make_dit_model_input_params_from_legacy(
    xllm::ModelInputParams&& src);
void apply_dit_model_input_params_to_legacy(const DitModelInputParams& src,
                                            xllm::ModelInputParams* dst);

RecModelInputParams make_rec_model_input_params_from_legacy(
    const xllm::ModelInputParams& src);
RecModelInputParams make_rec_model_input_params_from_legacy(
    xllm::ModelInputParams&& src);
void apply_rec_model_input_params_to_legacy(const RecModelInputParams& src,
                                            xllm::ModelInputParams* dst);

ModelInputParamBundle make_model_input_param_bundle_from_legacy(
    const xllm::ModelInputParams& src);
ModelInputParamBundle make_model_input_param_bundle_from_legacy(
    xllm::ModelInputParams&& src);
void apply_model_input_param_bundle_to_legacy(const ModelInputParamBundle& src,
                                              xllm::ModelInputParams* dst);

}  // namespace model_input
}  // namespace xllm
