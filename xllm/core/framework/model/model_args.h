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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"

namespace xllm {

struct ModelArgs {
  PROPERTY(std::string, model_type);

  PROPERTY(std::string, dtype);

  PROPERTY(int64_t, hidden_size) = 0;

  PROPERTY(std::string, hidden_act);

  // intermediate size
  PROPERTY(int64_t, intermediate_size) = 0;

  PROPERTY(int64_t, n_layers) = 0;

  // attn head dim
  PROPERTY(int64_t, head_dim) = 0;

  // attn head num
  PROPERTY(int64_t, n_heads) = 0;

  // attn head num for key/value
  PROPERTY(std::optional<int64_t>, n_kv_heads);

  PROPERTY(int64_t, vocab_size) = -1;

  PROPERTY(float, rms_norm_eps) = 0.0f;

  PROPERTY(float, layer_norm_eps) = 0.0f;

  PROPERTY(int64_t, rotary_dim) = 0;

  // the base period of the rotary position embeddings.
  PROPERTY(float, rope_theta) = 10000.0f;

  // rope_scaling related args
  PROPERTY(std::string, rope_scaling_rope_type);
  PROPERTY(float, rope_scaling_factor) = 0.0f;
  PROPERTY(float, rope_scaling_low_freq_factor) = 0.0f;
  PROPERTY(float, rope_scaling_high_freq_factor) = 0.0f;
  PROPERTY(float, rope_extrapolation_factor) = 0.0f;
  PROPERTY(int64_t, rope_scaling_original_max_position_embeddings) = 0;
  PROPERTY(int64_t, rope_scaling_beta_fast) = 0;
  PROPERTY(int64_t, rope_scaling_beta_slow) = 0;
  PROPERTY(float, rope_scaling_attn_factor) = 0.0f;
  PROPERTY(float, rope_scaling_mscale) = 0.0f;
  PROPERTY(float, rope_scaling_mscale_all_dim) = 0.0f;
  PROPERTY(std::vector<int64_t>, rope_scaling_mrope_section);

  // the maximum sequence length to use for rotary position embeddings.
  PROPERTY(int64_t, max_position_embeddings) = 0;

  // token id for beginning of sentence.
  PROPERTY(int32_t, bos_token_id) = 0;

  // token id for end of sentence.
  PROPERTY(int32_t, eos_token_id) = 0;

  // token id vector for end of sentence.
  PROPERTY(std::vector<int32_t>, eos_token_id_vec);

  // token id for pad of sentence.
  PROPERTY(int32_t, pad_token_id) = 0;

  // scaling factor used on the attention scores
  PROPERTY(std::optional<float>, attn_scalar);

  // whether to use bias. only used for mpt models
  PROPERTY(bool, no_bias) = false;

  // whether to use bias for qkv.
  PROPERTY(bool, qkv_bias) = false;

  // Stop token ids for decoding.
  PROPERTY(std::unordered_set<int32_t>, stop_token_ids);

  // deepseek v2/v3
  PROPERTY(int32_t, first_k_dense_replace) = 0;
  PROPERTY(int32_t, moe_layer_freq) = 0;
  // deepseek v2/v3 MoE
  PROPERTY(std::string, topk_method);
  PROPERTY(int32_t, n_routed_experts) = 0;
  PROPERTY(int32_t, n_shared_experts) = 0;
  PROPERTY(int32_t, num_experts_per_tok) = 0;
  PROPERTY(int32_t, moe_intermediate_size) = 0;
  PROPERTY(float, routed_scaling_factor) = 0.0f;
  PROPERTY(bool, norm_topk_prob) = false;
  PROPERTY(int32_t, n_group) = 0;
  PROPERTY(int32_t, topk_group) = 0;
  // deepseek v2/v3 MLA
  PROPERTY(int32_t, qk_nope_head_dim) = 0;
  PROPERTY(int32_t, qk_rope_head_dim) = 0;
  PROPERTY(int32_t, v_head_dim) = 0;
  PROPERTY(int32_t, q_lora_rank) = 0;
  PROPERTY(int32_t, kv_lora_rank) = 0;

  PROPERTY(int32_t, vision_start_token_id) = 0;
  PROPERTY(int32_t, vision_end_token_id) = 0;
  PROPERTY(int32_t, vision_token_id) = 0;
  PROPERTY(int32_t, image_token_id) = 0;
  PROPERTY(int32_t, video_token_id) = 0;

  PROPERTY(std::string, vision_custom_adapter);
  PROPERTY(int32_t, vision_max_slice_nums) = 0;

  // qwen3 moe
  PROPERTY(bool, attention_bias) = false;
  PROPERTY(float, attention_dropout) = 0.0f;
  PROPERTY(int32_t, decoder_sparse_step) = 1;
  PROPERTY(float, initializer_range) = 0.02f;
  PROPERTY(std::vector<int32_t>, mlp_only_layers) = {};
  PROPERTY(int64_t, num_attention_heads) = 32;
  PROPERTY(int32_t, num_experts) = 128;
  PROPERTY(bool, output_router_logits) = false;
  PROPERTY(int32_t, rope_scaling) = -1;
  PROPERTY(float, router_aux_loss_coef) = 0.001f;

  // Vision model's dropout
  PROPERTY(float, mm_dropout) = 0.0f;

  // Vision model's hidden_act
  PROPERTY(std::string, mm_hidden_act);

  // Vision model's mm_hidden_size
  PROPERTY(int64_t, mm_hidden_size) = 0;

  // Vision model's mm_image_size
  PROPERTY(int64_t, mm_image_size) = 0;

  // Vision model's mm_intermediate_size
  PROPERTY(int64_t, mm_intermediate_size) = 0;

  // Vision model's mm_num_channels
  PROPERTY(int64_t, mm_num_channels) = 0;

  // Vision model's mm_initializer_range
  PROPERTY(float, mm_initializer_range) = 0.0f;

  // Vision model's mm_layer_norm_eps
  PROPERTY(float, mm_layer_norm_eps) = 0;

  // Vision model's mm_num_attention_heads
  PROPERTY(int64_t, mm_num_attention_heads) = 0;

  // Vision model's mm_num_beam_groups
  PROPERTY(int64_t, mm_num_beam_groups) = 0;

  // Vision model's mm_num_beams
  PROPERTY(int64_t, mm_num_beams) = 0;

  // Vision model's mm_num_hidden_layers
  PROPERTY(int64_t, mm_num_hidden_layers) = 0;

  // Vision model's mm_num_return_sequences
  PROPERTY(int64_t, mm_num_return_sequences) = 0;

  // Vision model's mm_output_attentions
  PROPERTY(bool, mm_output_attentions) = false;

  // Vision model's mm_output_hidden_states
  PROPERTY(bool, mm_output_hidden_states) = false;

  // Vision model's mm_output_scores
  PROPERTY(bool, mm_output_scores) = false;

  // Vision model's mm_patch_size
  PROPERTY(int64_t, mm_patch_size) = 0;

  // Vision model's mm_projection_dim
  PROPERTY(int64_t, mm_projection_dim) = 0;

  PROPERTY(int64_t, mm_spatial_merge_size) = 0;
  PROPERTY(int64_t, mm_spatial_patch_size) = 0;

  // Vision model's mm_remove_invalid_values
  PROPERTY(bool, mm_remove_invalid_values) = false;

  // Vision model's mm_repetition_penalty
  PROPERTY(float, mm_repetition_penalty) = 0.0f;

  // Vision model's mm_return_dict
  PROPERTY(bool, mm_return_dict) = false;

  // Vision model's mm_return_dict_in_generate
  PROPERTY(bool, mm_return_dict_in_generate) = false;

  // Vision model's mm_temperature
  PROPERTY(float, mm_temperature) = 0.0f;

  // Vision model's mm_tie_encoder_decoder
  PROPERTY(bool, mm_tie_encoder_decoder) = false;

  // Vision model's mm_tie_word_embeddings
  PROPERTY(bool, mm_tie_word_embeddings) = false;

  // Vision model's mm_top_k
  PROPERTY(int64_t, mm_top_k) = 0;

  // Vision model's mm_top_p
  PROPERTY(float, mm_top_p) = 0.0f;

  // Vision model's mm_torchscript
  PROPERTY(bool, mm_torchscript) = false;

  // Vision model's mm_use_bfloat16
  PROPERTY(bool, mm_use_bfloat16) = false;

  // Vision model's mm_head_dim
  PROPERTY(int64_t, mm_head_dim) = 0;

  // Vision model's mm_vocab_size
  PROPERTY(int64_t, mm_vocab_size) = 0;

  PROPERTY(int, mm_window_size) = 0;
  PROPERTY(std::vector<int64_t>, mm_fullatt_block_indexes);
  PROPERTY(std::vector<int64_t>, mm_deepstack_visual_indexes);
  PROPERTY(int, mm_tokens_per_second) = 0;
  PROPERTY(int, mm_temporal_patch_size) = 0;

  // VLM model projector's mm_projector_type
  PROPERTY(std::string, mm_projector_type);

  //
  PROPERTY(int64_t, mm_num_position_embeddings);
  // VLM model projector's mm_projector_hidden_act
  PROPERTY(std::string, mm_projector_hidden_act);

  // VLM model projector's mm_projector_n_layers
  PROPERTY(int64_t, mm_projector_n_layers) = 0;

  // VLM model projector's mm_vision_feature_layer
  PROPERTY(int64_t, mm_vision_feature_layer) = 0;

  // VLM model projector's mm_vision_feature_select_strategy
  PROPERTY(std::string, mm_vision_feature_select_strategy);

  // mm image begin
  // VLM image preprocessor centor crop
  PROPERTY(bool, mm_image_do_center_crop) = false;
  PROPERTY(int, mm_image_crop_height_size) = 336;
  PROPERTY(int, mm_image_crop_width_size) = 336;

  // VLM image preprocessor resize
  PROPERTY(bool, mm_image_do_resize) = false;
  PROPERTY(int, mm_image_resize_shortest_edge) = 336;

  PROPERTY(int, mm_image_resample) = 0;

  // VLM image preprocessor resize
  PROPERTY(bool, mm_image_do_rescale) = false;
  PROPERTY(double, mm_image_rescale_factor) = 0;

  // VLM image preprocessor normalization
  PROPERTY(bool, mm_image_do_normalize) = false;
  PROPERTY(std::vector<double>, mm_image_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_image_normalize_std) = {};

  PROPERTY(int, mm_image_min_pixels) = 0;
  PROPERTY(int, mm_image_max_pixels) = 0;

  PROPERTY(int64_t, mm_image_shortest_edge) = 0;
  PROPERTY(int64_t, mm_image_longest_edge) = 0;

  PROPERTY(int, mm_image_patch_size) = 0;
  PROPERTY(int, mm_image_temporal_patch_size) = 0;
  PROPERTY(int, mm_image_merge_size) = 0;

  PROPERTY(int, mm_image_feature_size) = 0;
  PROPERTY(int, mm_scale_resolution) = 0;
  PROPERTY(bool, mm_slice_mode) = false;
  PROPERTY(bool, mm_use_image_id) = false;

  // mm image end

  PROPERTY(int64_t, mm_image_token_index) = 0;
  PROPERTY(int64_t, mm_pad_token_id) = 0;

  // whether to tie weight embeddings
  PROPERTY(bool, tie_word_embeddings) = false;

  // sliding window for attention
  PROPERTY(bool, use_sliding_window) = false;
  PROPERTY(int32_t, sliding_window) = -1;
  PROPERTY(int32_t, max_window_layers) = 0;

  PROPERTY(int32_t, query_num) = 0;
  PROPERTY(bool, image_embedding_mode) = false;

  // number of speculative decoding tokens
  PROPERTY(int64_t, num_speculative_tokens) = 0;

  // VAE related args
  PROPERTY(int64_t, in_channels) = -1;
  PROPERTY(int64_t, out_channels) = -1;
  PROPERTY(std::vector<std::string>, down_block_types) = {

  };
  PROPERTY(std::vector<std::string>, up_block_types) = {

  };
  PROPERTY(std::vector<int64_t>, block_out_channels) = {};
  PROPERTY(int64_t, layers_per_block) = 1;
  PROPERTY(int64_t, latent_channels) = -1;
  PROPERTY(int64_t, norm_num_groups) = -1;
  PROPERTY(int64_t, sample_size) = -1;
  PROPERTY(float, scale_factor) = 0.0f;
  PROPERTY(float, shift_factor) = 0.0f;
  PROPERTY(bool, mid_block_add_attention) = true;
  PROPERTY(bool, force_upcast) = true;
  PROPERTY(bool, use_quant_conv) = false;
  PROPERTY(bool, use_post_quant_conv) = false;

  // dit related args
  PROPERTY(int64_t, joint_attention_dim) = 0;
  PROPERTY(int64_t, pooled_projection_dim) = 0;
  PROPERTY(bool, guidance_embeds) = true;
  PROPERTY(std::vector<int64_t>, axes_dims_rope) = {};
  PROPERTY(int64_t, num_single_layers) = 0;

  // rec related args
  PROPERTY(int64_t, d_model) = 0;
  PROPERTY(int64_t, num_layers) = 0;
  PROPERTY(int64_t, d_kv) = 0;
  PROPERTY(int64_t, d_ff) = 0;
  PROPERTY(std::string, act_fn) = "";
  PROPERTY(bool, is_gated_act) = true;
  PROPERTY(int64_t, relative_attention_num_buckets) = 0;
  PROPERTY(int64_t, relative_attention_max_distance) = 0;

  PROPERTY(int64_t, n_encoder_layers) = 0;
  PROPERTY(int64_t, decoder_head_dim) = 0;
  PROPERTY(int64_t, decoder_n_heads) = 0;
  PROPERTY(std::optional<int64_t>, decoder_n_kv_heads);
  PROPERTY(bool, use_absolute_position_embedding) = false;

  PROPERTY(bool, use_moe) = false;
  PROPERTY(std::string, moe_score_func);
  PROPERTY(float, moe_route_scale) = 1.0f;
  PROPERTY(bool, moe_use_shared_experts) = false;

  // scheduler related args
  PROPERTY(int64_t, num_train_timesteps) = 0;
  PROPERTY(int64_t, shift) = 0;
  PROPERTY(bool, use_dynamic_shifting) = false;
  PROPERTY(float, base_shift) = 0;
  PROPERTY(float, max_shift) = 0;
  PROPERTY(int64_t, base_image_seq_len) = 0;
  PROPERTY(int64_t, max_image_seq_len) = 0;
};

inline std::ostream& operator<<(std::ostream& os, const ModelArgs& args) {
  os << "ModelArgs: [model_type: " << args.model_type();
  os << ", image_embedding_mode: " << args.image_embedding_mode();
  os << ", dtype: " << args.dtype();
  os << ", hidden_size: " << args.hidden_size();
  os << ", hidden_act: " << args.hidden_act();
  os << ", intermediate_size: " << args.intermediate_size();
  os << ", n_layers: " << args.n_layers();
  os << ", head_dim: " << args.head_dim();
  os << ", n_heads: " << args.n_heads();
  os << ", n_kv_heads: " << args.n_kv_heads().value_or(-1);
  os << ", vocab_size: " << args.vocab_size();
  os << ", rms_norm_eps: " << args.rms_norm_eps();
  os << ", layer_norm_eps: " << args.layer_norm_eps();
  os << ", rotary_dim: " << args.rotary_dim();
  os << ", rope_theta: " << args.rope_theta();
  os << ", rope_scaling_rope_type: " << args.rope_scaling_rope_type();
  os << ", rope_scaling_factor: " << args.rope_scaling_factor();
  os << ", rope_scaling_low_freq_factor: "
     << args.rope_scaling_low_freq_factor();
  os << ", rope_scaling_high_freq_factor: "
     << args.rope_scaling_high_freq_factor();
  os << ", rope_scaling_original_max_position_embeddings: "
     << args.rope_scaling_original_max_position_embeddings();
  os << ", rope_scaling_mrope_section: [";
  for (const auto& sec : args.rope_scaling_mrope_section()) {
    os << sec << ", ";
  }
  os << "]";
  os << ", max_position_embeddings: " << args.max_position_embeddings();
  os << ", bos_token_id: " << args.bos_token_id();
  os << ", eos_token_id: " << args.eos_token_id();
  os << ", pad_token_id: " << args.pad_token_id();
  os << ", attn_scalar: " << args.attn_scalar().value_or(0.0f);
  os << ", no_bias: " << args.no_bias();
  os << ", qkv_bias: " << args.qkv_bias();
  os << ", stop_token_ids: [";
  for (const auto& id : args.stop_token_ids()) {
    os << id << ", ";
  }
  os << "]";
  os << ", vision_start_token_id: " << args.vision_start_token_id();
  os << ", vision_end_token_id: " << args.vision_end_token_id();
  os << ", vision_token_id: " << args.vision_token_id();
  os << ", image_token_id: " << args.image_token_id();
  os << ", video_token_id: " << args.video_token_id();
  os << ", vision_custom_adapter: " << args.vision_custom_adapter();
  os << ", vision_max_slice_nums: " << args.vision_max_slice_nums();
  os << ", mm_dropout: " << args.mm_dropout();
  os << ", mm_hidden_act: " << args.mm_hidden_act();
  os << ", mm_hidden_size: " << args.mm_hidden_size();
  os << ", mm_image_size: " << args.mm_image_size();
  os << ", mm_intermediate_size: " << args.mm_intermediate_size();
  os << ", mm_num_channels: " << args.mm_num_channels();
  os << ", mm_initializer_range: " << args.mm_initializer_range();
  os << ", mm_layer_norm_eps: " << args.mm_layer_norm_eps();
  os << ", mm_num_attention_heads: " << args.mm_num_attention_heads();
  os << ", mm_num_beam_groups: " << args.mm_num_beam_groups();
  os << ", mm_num_beams: " << args.mm_num_beams();
  os << ", mm_num_hidden_layers: " << args.mm_num_hidden_layers();
  os << ", mm_num_return_sequences: " << args.mm_num_return_sequences();
  os << ", mm_output_attentions: " << args.mm_output_attentions();
  os << ", mm_output_hidden_states: " << args.mm_output_hidden_states();
  os << ", mm_output_scores: " << args.mm_output_scores();
  os << ", mm_patch_size: " << args.mm_patch_size();
  os << ", mm_projection_dim: " << args.mm_projection_dim();
  os << ", mm_spatial_merge_size: " << args.mm_spatial_merge_size();
  os << ", mm_spatial_patch_size: " << args.mm_spatial_patch_size();
  os << ", mm_remove_invalid_values: " << args.mm_remove_invalid_values();
  os << ", mm_repetition_penalty: " << args.mm_repetition_penalty();
  os << ", mm_return_dict: " << args.mm_return_dict();
  os << ", mm_return_dict_in_generate: " << args.mm_return_dict_in_generate();
  os << ", mm_temperature: " << args.mm_temperature();
  os << ", mm_tie_encoder_decoder: " << args.mm_tie_encoder_decoder();
  os << ", mm_tie_word_embeddings: " << args.mm_tie_word_embeddings();
  os << ", mm_top_k: " << args.mm_top_k();
  os << ", mm_top_p: " << args.mm_top_p();
  os << ", mm_torchscript: " << args.mm_torchscript();
  os << ", mm_use_bfloat16: " << args.mm_use_bfloat16();
  os << ", mm_head_dim: " << args.mm_head_dim();
  os << ", mm_vocab_size: " << args.mm_vocab_size();
  os << ", mm_window_size: " << args.mm_window_size();
  os << ", mm_fullatt_block_indexes: [";
  for (auto& index : args.mm_fullatt_block_indexes()) {
    os << index << ",";
  }
  os << "]";
  os << ", mm_deepstack_visual_indexes: [";
  for (auto& index : args.mm_deepstack_visual_indexes()) {
    os << index << ",";
  }
  os << "]";
  os << ", mm_tokens_per_second: " << args.mm_tokens_per_second();
  os << ", mm_temporal_patch_size: " << args.mm_temporal_patch_size();
  os << ", mm_projector_type: " << args.mm_projector_type();
  os << ", mm_projector_hidden_act: " << args.mm_projector_hidden_act();
  os << ", mm_projector_n_layers: " << args.mm_projector_n_layers();
  os << ", mm_vision_feature_layer: " << args.mm_vision_feature_layer();
  os << ", mm_vision_feature_select_strategy: "
     << args.mm_vision_feature_select_strategy();
  os << ", mm_image_do_center_crop: " << args.mm_image_do_center_crop();
  os << ", mm_image_crop_height_size: " << args.mm_image_crop_height_size();
  os << ", mm_image_crop_width_size: " << args.mm_image_crop_width_size();
  os << ", mm_image_do_resize: " << args.mm_image_do_resize();
  os << ", mm_image_resize_shortest_edge: "
     << args.mm_image_resize_shortest_edge();
  os << ", mm_image_resample: " << args.mm_image_resample();
  os << ", mm_image_do_rescale: " << args.mm_image_do_rescale();
  os << ", mm_image_rescale_factor: " << args.mm_image_rescale_factor();
  os << ", mm_image_do_normalize: " << args.mm_image_do_normalize();
  os << ", mm_image_normalize_mean: [";
  for (const auto& mean : args.mm_image_normalize_mean()) {
    os << mean << ", ";
  }
  os << "], mm_image_normalize_std: [";
  for (const auto& std : args.mm_image_normalize_std()) {
    os << std << ", ";
  }
  os << "]";
  os << ", mm_image_shortest_edge: " << args.mm_image_shortest_edge();
  os << ", mm_image_longest_edge: " << args.mm_image_longest_edge();
  os << ", mm_image_min_pixels: " << args.mm_image_min_pixels();
  os << ", mm_image_max_pixels: " << args.mm_image_max_pixels();
  os << ", mm_image_patch_size: " << args.mm_image_patch_size();
  os << ", mm_image_temporal_patch_size: "
     << args.mm_image_temporal_patch_size();
  os << ", mm_image_merge_size: " << args.mm_image_merge_size();
  os << ", mm_image_token_index: " << args.mm_image_token_index();
  os << ", mm_pad_token_id: " << args.mm_pad_token_id();
  os << ", tie_word_embeddings: " << args.tie_word_embeddings();
  os << ", use_sliding_window: " << args.use_sliding_window();
  os << ", sliding_window: " << args.sliding_window();
  os << ", max_window_layers: " << args.max_window_layers();
  os << ", query_num: " << args.query_num();
  os << ", num_speculative_tokens: " << args.num_speculative_tokens();
  os << ", in_channels: " << args.in_channels();
  os << ", out_channels: " << args.out_channels();
  os << ", down_block_types: [";
  for (const auto& type : args.down_block_types()) {
    os << type << ", ";
  }
  os << "]";
  os << ", up_block_types: [";
  for (const auto& type : args.up_block_types()) {
    os << type << ", ";
  }
  os << "]";
  os << ", block_out_channels: [";
  for (const auto& channel : args.block_out_channels()) {
    os << channel << ", ";
  }
  os << "]";
  os << ", layers_per_block: " << args.layers_per_block();
  os << ", latent_channels: " << args.latent_channels();
  os << ", norm_num_groups: " << args.norm_num_groups();
  os << ", sample_size: " << args.sample_size();
  os << ", scale_factor: " << args.scale_factor();
  os << ", shift_factor: " << args.shift_factor();
  os << ", mid_block_add_attention: " << args.mid_block_add_attention();
  os << ", force_upcast: " << args.force_upcast();
  os << ", use_quant_conv: " << args.use_quant_conv();
  os << ", use_post_quant_conv: " << args.use_post_quant_conv();
  os << ", joint_attention_dim: " << args.joint_attention_dim();
  os << ", pooled_projection_dim: " << args.pooled_projection_dim();
  os << ", guidance_embeds: " << args.guidance_embeds();
  os << ", axes_dims_rope: [";
  for (const auto& dim : args.axes_dims_rope()) {
    os << dim << ", ";
  }
  os << "]";
  os << ", num_single_layers: " << args.num_single_layers();
  os << ", d_model: " << args.d_model();
  os << ", num_layers: " << args.num_layers();
  os << ", d_kv: " << args.d_kv();
  os << ", d_ff: " << args.d_ff();
  os << ", act_fn: " << args.act_fn();
  os << ", is_gated_act: " << args.is_gated_act();
  os << ", relative_attention_num_buckets: "
     << args.relative_attention_num_buckets();
  os << ", relative_attention_max_distance: "
     << args.relative_attention_max_distance();
  os << ", num_train_timesteps: " << args.num_train_timesteps();
  os << ", shift: " << args.shift();
  os << ", use_dynamic_shifting: " << args.use_dynamic_shifting();
  os << ", base_shift: " << args.base_shift();
  os << ", max_shift: " << args.max_shift();
  os << ", base_image_seq_len: " << args.base_image_seq_len();
  os << ", max_image_seq_len: " << args.max_image_seq_len();
  os << "]";
  return os;
}

}  // namespace xllm
