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

#include <variant>

namespace xllm {
namespace model_input {

LLMModelInputParams LLMModelInputParams::from_legacy(
    const xllm::ModelInputParams& src) {
  LLMModelInputParams dst;
  dst.enable_mla = src.enable_mla;
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

void LLMModelInputParams::apply_to_legacy(xllm::ModelInputParams* dst) const {
  dst->enable_mla = enable_mla;
  dst->batch_forward_type = batch_forward_type;
  dst->num_sequences = num_sequences;
  dst->kv_max_seq_len = kv_max_seq_len;
  dst->q_max_seq_len = q_max_seq_len;
  dst->batch_id = batch_id;
  dst->q_seq_lens = q_seq_lens;
  dst->kv_seq_lens = kv_seq_lens;
  dst->q_cu_seq_lens = q_cu_seq_lens;
  dst->kv_seq_lens_vec = kv_seq_lens_vec;
  dst->q_seq_lens_vec = q_seq_lens_vec;
  dst->new_cache_slots = new_cache_slots;
  dst->block_tables = block_tables;
  dst->paged_kv_indptr = paged_kv_indptr;
  dst->paged_kv_indices = paged_kv_indices;
  dst->paged_kv_last_page_len = paged_kv_last_page_len;
  dst->new_cache_slot_offsets = new_cache_slot_offsets;
  dst->kv_cache_start_offsets = kv_cache_start_offsets;
  dst->input_embedding = input_embedding;
  dst->dp_global_token_nums = dp_global_token_nums;
  dst->dp_is_decode = dp_is_decode;
  dst->embedding_ids = embedding_ids;
  dst->request_ids = request_ids;
  dst->extra_token_ids = extra_token_ids;
  dst->swap_blocks = swap_blocks;
  dst->src_block_indices = src_block_indices;
  dst->dst_block_indices = dst_block_indices;
  dst->cum_sum = cum_sum;
#if defined(USE_NPU)
  dst->layer_synchronizer = layer_synchronizer;
  dst->layers_per_bacth_copy = layers_per_bacth_copy;
  dst->layer_wise_load_synchronizer = layer_wise_load_synchronizer;
#endif
  dst->dp_ep_padding_data = dp_ep_padding_data;
  dst->cp_ep_padding_data = cp_ep_padding_data;
  dst->cp_prefill_inputs = cp_prefill_inputs;
  dst->expert_load_data = expert_load_data;
  dst->expert_array = expert_array;
  dst->kv_cache_tokens_nums = kv_cache_tokens_nums;
  dst->kv_cache_tokens_nums_host = kv_cache_tokens_nums_host;
  dst->history_compressed_kv = history_compressed_kv;
  dst->history_k_rope = history_k_rope;
  dst->ring_cur_seqlen = ring_cur_seqlen;
  dst->ring_cur_seqlen_host = ring_cur_seqlen_host;
  dst->ring_cache_seqlen = ring_cache_seqlen;
  dst->ring_cache_seqlen_host = ring_cache_seqlen_host;
  dst->graph_buffer.attn_mask = graph_attn_mask;
  dst->graph_buffer.tiling_data = graph_tiling_data;
  dst->attn_metadata = attn_metadata;
  dst->enable_cuda_graph = enable_cuda_graph;
}

VLMModelInputParams VLMModelInputParams::from_legacy(
    const xllm::ModelInputParams& src) {
  VLMModelInputParams dst;
  dst.mm_data = src.mm_data;
  dst.deep_stacks = src.deep_stacks;
  dst.visual_pos_masks = src.visual_pos_masks;
  return dst;
}

void VLMModelInputParams::apply_to_legacy(xllm::ModelInputParams* dst) const {
  dst->mm_data = mm_data;
  dst->deep_stacks = deep_stacks;
  dst->visual_pos_masks = visual_pos_masks;
}

DitModelInputParams DitModelInputParams::from_legacy(
    const xllm::ModelInputParams& src) {
  DitModelInputParams dst;
  dst.dit_forward_input = src.dit_forward_input;
  return dst;
}

void DitModelInputParams::apply_to_legacy(xllm::ModelInputParams* dst) const {
  dst->dit_forward_input = dit_forward_input;
}

RecModelInputParams RecModelInputParams::from_legacy(
    const xllm::ModelInputParams& src) {
  RecModelInputParams dst;
  dst.rec_params = src.rec_params;
  return dst;
}

void RecModelInputParams::apply_to_legacy(xllm::ModelInputParams* dst) const {
  dst->rec_params = rec_params;
}

ModelInputParamBundle ModelInputParamBundle::from_legacy(
    const xllm::ModelInputParams& src) {
  ModelInputParamBundle bundle;
  bundle.llm = std::make_shared<LLMModelInputParams>(
      LLMModelInputParams::from_legacy(src));
  if (src.mm_data.valid() || !src.deep_stacks.empty() ||
      src.visual_pos_masks.defined()) {
    bundle.vlm = std::make_shared<VLMModelInputParams>(
        VLMModelInputParams::from_legacy(src));
  }
  if (src.dit_forward_input.valid()) {
    bundle.dit = std::make_shared<DitModelInputParams>(
        DitModelInputParams::from_legacy(src));
  }
  if (!std::holds_alternative<std::monostate>(src.rec_params)) {
    bundle.rec = std::make_shared<RecModelInputParams>(
        RecModelInputParams::from_legacy(src));
  }
  return bundle;
}

void ModelInputParamBundle::apply_to_legacy(xllm::ModelInputParams* dst) const {
  if (llm != nullptr) {
    llm->apply_to_legacy(dst);
  }
  if (vlm != nullptr) {
    vlm->apply_to_legacy(dst);
  }
  if (dit != nullptr) {
    dit->apply_to_legacy(dst);
  }
  if (rec != nullptr) {
    rec->apply_to_legacy(dst);
  }
}

}  // namespace model_input
}  // namespace xllm
