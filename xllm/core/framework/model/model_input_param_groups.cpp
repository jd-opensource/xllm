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

#include "model_input_param_groups.h"

#include <utility>
#include <variant>

namespace xllm {
namespace model_input {

LLMModelInputParams make_llm_model_input_params_from_legacy(
    const xllm::ModelInputParams& src) {
  LLMModelInputParams dst;
  dst.batch_forward_type = src.batch_forward_type;
  dst.num_sequences = src.num_sequences;
  dst.kv_max_seq_len = src.kv_max_seq_len;
  dst.q_max_seq_len = src.q_max_seq_len;
  dst.batch_id = src.batch_id;
  dst.q_seq_lens = src.q_seq_lens;
  dst.kv_seq_lens = src.kv_seq_lens;
  dst.q_cu_seq_lens = src.q_cu_seq_lens;
  dst.kv_seq_lens_vec = src.kv_seq_lens_vec;
  dst.q_seq_lens_vec = src.q_seq_lens_vec;
  dst.new_cache_slots = src.new_cache_slots;
  dst.block_tables = src.block_tables;
  dst.paged_kv_indptr = src.paged_kv_indptr;
  dst.paged_kv_indices = src.paged_kv_indices;
  dst.paged_kv_last_page_len = src.paged_kv_last_page_len;
  dst.new_cache_slot_offsets = src.new_cache_slot_offsets;
  dst.kv_cache_start_offsets = src.kv_cache_start_offsets;
  dst.input_embedding = src.input_embedding;
  dst.dp_global_token_nums = src.dp_global_token_nums;
  dst.dp_is_decode = src.dp_is_decode;
  dst.embedding_ids = src.embedding_ids;
  dst.request_ids = src.request_ids;
  dst.extra_token_ids = src.extra_token_ids;
  dst.swap_blocks = src.swap_blocks;
  dst.src_block_indices = src.src_block_indices;
  dst.dst_block_indices = src.dst_block_indices;
  dst.cum_sum = src.cum_sum;
#if defined(USE_NPU)
  dst.layer_synchronizer = src.layer_synchronizer;
  dst.layers_per_bacth_copy = src.layers_per_bacth_copy;
  dst.layer_wise_load_synchronizer = src.layer_wise_load_synchronizer;
#endif
  dst.dp_ep_padding_data = src.dp_ep_padding_data;
  dst.cp_ep_padding_data = src.cp_ep_padding_data;
  dst.cp_prefill_inputs = src.cp_prefill_inputs;
  dst.expert_load_data = src.expert_load_data;
  dst.expert_array = src.expert_array;
  dst.kv_cache_tokens_nums = src.kv_cache_tokens_nums;
  dst.kv_cache_tokens_nums_host = src.kv_cache_tokens_nums_host;
  dst.history_compressed_kv = src.history_compressed_kv;
  dst.history_k_rope = src.history_k_rope;
  dst.ring_cur_seqlen = src.ring_cur_seqlen;
  dst.ring_cur_seqlen_host = src.ring_cur_seqlen_host;
  dst.ring_cache_seqlen = src.ring_cache_seqlen;
  dst.ring_cache_seqlen_host = src.ring_cache_seqlen_host;
  dst.graph_attn_mask = src.graph_buffer.attn_mask;
  dst.graph_tiling_data = src.graph_buffer.tiling_data;
  dst.attn_metadata = src.attn_metadata;
  dst.enable_cuda_graph = src.enable_cuda_graph;
  return dst;
}

LLMModelInputParams make_llm_model_input_params_from_legacy(
    xllm::ModelInputParams&& src) {
  LLMModelInputParams dst;
  dst.batch_forward_type = src.batch_forward_type;
  dst.num_sequences = src.num_sequences;
  dst.kv_max_seq_len = src.kv_max_seq_len;
  dst.q_max_seq_len = src.q_max_seq_len;
  dst.batch_id = src.batch_id;
  dst.q_seq_lens = std::move(src.q_seq_lens);
  dst.kv_seq_lens = std::move(src.kv_seq_lens);
  dst.q_cu_seq_lens = std::move(src.q_cu_seq_lens);
  dst.kv_seq_lens_vec = std::move(src.kv_seq_lens_vec);
  dst.q_seq_lens_vec = std::move(src.q_seq_lens_vec);
  dst.new_cache_slots = std::move(src.new_cache_slots);
  dst.block_tables = std::move(src.block_tables);
  dst.paged_kv_indptr = std::move(src.paged_kv_indptr);
  dst.paged_kv_indices = std::move(src.paged_kv_indices);
  dst.paged_kv_last_page_len = std::move(src.paged_kv_last_page_len);
  dst.new_cache_slot_offsets = std::move(src.new_cache_slot_offsets);
  dst.kv_cache_start_offsets = std::move(src.kv_cache_start_offsets);
  dst.input_embedding = std::move(src.input_embedding);
  dst.dp_global_token_nums = std::move(src.dp_global_token_nums);
  dst.dp_is_decode = std::move(src.dp_is_decode);
  dst.embedding_ids = std::move(src.embedding_ids);
  dst.request_ids = std::move(src.request_ids);
  dst.extra_token_ids = std::move(src.extra_token_ids);
  dst.swap_blocks = std::move(src.swap_blocks);
  dst.src_block_indices = std::move(src.src_block_indices);
  dst.dst_block_indices = std::move(src.dst_block_indices);
  dst.cum_sum = std::move(src.cum_sum);
#if defined(USE_NPU)
  dst.layer_synchronizer = std::move(src.layer_synchronizer);
  dst.layers_per_bacth_copy = src.layers_per_bacth_copy;
  dst.layer_wise_load_synchronizer =
      std::move(src.layer_wise_load_synchronizer);
#endif
  dst.dp_ep_padding_data = std::move(src.dp_ep_padding_data);
  dst.cp_ep_padding_data = std::move(src.cp_ep_padding_data);
  dst.cp_prefill_inputs = std::move(src.cp_prefill_inputs);
  dst.expert_load_data = std::move(src.expert_load_data);
  dst.expert_array = std::move(src.expert_array);
  dst.kv_cache_tokens_nums = std::move(src.kv_cache_tokens_nums);
  dst.kv_cache_tokens_nums_host = std::move(src.kv_cache_tokens_nums_host);
  dst.history_compressed_kv = std::move(src.history_compressed_kv);
  dst.history_k_rope = std::move(src.history_k_rope);
  dst.ring_cur_seqlen = std::move(src.ring_cur_seqlen);
  dst.ring_cur_seqlen_host = std::move(src.ring_cur_seqlen_host);
  dst.ring_cache_seqlen = std::move(src.ring_cache_seqlen);
  dst.ring_cache_seqlen_host = std::move(src.ring_cache_seqlen_host);
  dst.graph_attn_mask = std::move(src.graph_buffer.attn_mask);
  dst.graph_tiling_data = std::move(src.graph_buffer.tiling_data);
  dst.attn_metadata = std::move(src.attn_metadata);
  dst.enable_cuda_graph = src.enable_cuda_graph;
  return dst;
}

void apply_llm_model_input_params_to_legacy(const LLMModelInputParams& src,
                                            xllm::ModelInputParams* dst) {
  dst->batch_forward_type = src.batch_forward_type;
  dst->num_sequences = src.num_sequences;
  dst->kv_max_seq_len = src.kv_max_seq_len;
  dst->q_max_seq_len = src.q_max_seq_len;
  dst->batch_id = src.batch_id;
  dst->q_seq_lens = src.q_seq_lens;
  dst->kv_seq_lens = src.kv_seq_lens;
  dst->q_cu_seq_lens = src.q_cu_seq_lens;
  dst->kv_seq_lens_vec = src.kv_seq_lens_vec;
  dst->q_seq_lens_vec = src.q_seq_lens_vec;
  dst->new_cache_slots = src.new_cache_slots;
  dst->block_tables = src.block_tables;
  dst->paged_kv_indptr = src.paged_kv_indptr;
  dst->paged_kv_indices = src.paged_kv_indices;
  dst->paged_kv_last_page_len = src.paged_kv_last_page_len;
  dst->new_cache_slot_offsets = src.new_cache_slot_offsets;
  dst->kv_cache_start_offsets = src.kv_cache_start_offsets;
  dst->input_embedding = src.input_embedding;
  dst->dp_global_token_nums = src.dp_global_token_nums;
  dst->dp_is_decode = src.dp_is_decode;
  dst->embedding_ids = src.embedding_ids;
  dst->request_ids = src.request_ids;
  dst->extra_token_ids = src.extra_token_ids;
  dst->swap_blocks = src.swap_blocks;
  dst->src_block_indices = src.src_block_indices;
  dst->dst_block_indices = src.dst_block_indices;
  dst->cum_sum = src.cum_sum;
#if defined(USE_NPU)
  dst->layer_synchronizer = src.layer_synchronizer;
  dst->layers_per_bacth_copy = src.layers_per_bacth_copy;
  dst->layer_wise_load_synchronizer = src.layer_wise_load_synchronizer;
#endif
  dst->dp_ep_padding_data = src.dp_ep_padding_data;
  dst->cp_ep_padding_data = src.cp_ep_padding_data;
  dst->cp_prefill_inputs = src.cp_prefill_inputs;
  dst->expert_load_data = src.expert_load_data;
  dst->expert_array = src.expert_array;
  dst->kv_cache_tokens_nums = src.kv_cache_tokens_nums;
  dst->kv_cache_tokens_nums_host = src.kv_cache_tokens_nums_host;
  dst->history_compressed_kv = src.history_compressed_kv;
  dst->history_k_rope = src.history_k_rope;
  dst->ring_cur_seqlen = src.ring_cur_seqlen;
  dst->ring_cur_seqlen_host = src.ring_cur_seqlen_host;
  dst->ring_cache_seqlen = src.ring_cache_seqlen;
  dst->ring_cache_seqlen_host = src.ring_cache_seqlen_host;
  dst->graph_buffer.attn_mask = src.graph_attn_mask;
  dst->graph_buffer.tiling_data = src.graph_tiling_data;
  dst->attn_metadata = src.attn_metadata;
  dst->enable_cuda_graph = src.enable_cuda_graph;
}

VLMModelInputParams make_vlm_model_input_params_from_legacy(
    const xllm::ModelInputParams& src) {
  VLMModelInputParams dst;
  dst.mm_data = src.mm_data;
  dst.deep_stacks = src.deep_stacks;
  dst.visual_pos_masks = src.visual_pos_masks;
  return dst;
}

VLMModelInputParams make_vlm_model_input_params_from_legacy(
    xllm::ModelInputParams&& src) {
  VLMModelInputParams dst;
  dst.mm_data = std::move(src.mm_data);
  dst.deep_stacks = std::move(src.deep_stacks);
  dst.visual_pos_masks = std::move(src.visual_pos_masks);
  return dst;
}

void apply_vlm_model_input_params_to_legacy(const VLMModelInputParams& src,
                                            xllm::ModelInputParams* dst) {
  dst->mm_data = src.mm_data;
  dst->deep_stacks = src.deep_stacks;
  dst->visual_pos_masks = src.visual_pos_masks;
}

DitModelInputParams make_dit_model_input_params_from_legacy(
    const xllm::ModelInputParams& src) {
  DitModelInputParams dst;
  dst.dit_forward_input = src.dit_forward_input;
  return dst;
}

DitModelInputParams make_dit_model_input_params_from_legacy(
    xllm::ModelInputParams&& src) {
  DitModelInputParams dst;
  dst.dit_forward_input = std::move(src.dit_forward_input);
  return dst;
}

void apply_dit_model_input_params_to_legacy(const DitModelInputParams& src,
                                            xllm::ModelInputParams* dst) {
  dst->dit_forward_input = src.dit_forward_input;
}

RecModelInputParams make_rec_model_input_params_from_legacy(
    const xllm::ModelInputParams& src) {
  RecModelInputParams dst;
  dst.rec_params = src.rec_params;
  return dst;
}

RecModelInputParams make_rec_model_input_params_from_legacy(
    xllm::ModelInputParams&& src) {
  RecModelInputParams dst;
  dst.rec_params = std::move(src.rec_params);
  return dst;
}

void apply_rec_model_input_params_to_legacy(const RecModelInputParams& src,
                                            xllm::ModelInputParams* dst) {
  dst->rec_params = src.rec_params;
}

ModelInputParamBundle make_model_input_param_bundle_from_legacy(
    const xllm::ModelInputParams& src) {
  ModelInputParamBundle bundle;
  bundle.llm = make_llm_model_input_params_from_legacy(src);
  if (src.mm_data.valid() || !src.deep_stacks.empty() ||
      src.visual_pos_masks.defined()) {
    bundle.vlm = make_vlm_model_input_params_from_legacy(src);
  }
  if (src.dit_forward_input.valid()) {
    bundle.dit = make_dit_model_input_params_from_legacy(src);
  }
  if (!std::holds_alternative<std::monostate>(src.rec_params)) {
    bundle.rec = make_rec_model_input_params_from_legacy(src);
  }
  return bundle;
}

ModelInputParamBundle make_model_input_param_bundle_from_legacy(
    xllm::ModelInputParams&& src) {
  ModelInputParamBundle bundle;
  bundle.llm = make_llm_model_input_params_from_legacy(std::move(src));
  if (src.mm_data.valid() || !src.deep_stacks.empty() ||
      src.visual_pos_masks.defined()) {
    bundle.vlm = make_vlm_model_input_params_from_legacy(std::move(src));
  }
  if (src.dit_forward_input.valid()) {
    bundle.dit = make_dit_model_input_params_from_legacy(std::move(src));
  }
  if (!std::holds_alternative<std::monostate>(src.rec_params)) {
    bundle.rec = make_rec_model_input_params_from_legacy(std::move(src));
  }
  return bundle;
}

void apply_model_input_param_bundle_to_legacy(const ModelInputParamBundle& src,
                                              xllm::ModelInputParams* dst) {
  if (src.llm.has_value()) {
    apply_llm_model_input_params_to_legacy(*(src.llm), dst);
  }
  if (src.vlm.has_value()) {
    apply_vlm_model_input_params_to_legacy(*(src.vlm), dst);
  }
  if (src.dit.has_value()) {
    apply_dit_model_input_params_to_legacy(*(src.dit), dst);
  }
  if (src.rec.has_value()) {
    apply_rec_model_input_params_to_legacy(*(src.rec), dst);
  }
}

}  // namespace model_input
}  // namespace xllm
