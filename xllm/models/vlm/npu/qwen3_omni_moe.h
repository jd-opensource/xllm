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

#include "qwen3_omni_moe_thinker.h"

namespace xllm {

class Qwen3OmniMoe_ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    thinker_ = register_module(
        "thinker", Qwen3OmniMoe_Thinker_ForConditionalGeneration(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    return thinker_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return thinker_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    thinker_->load_model(std::move(loader));
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    return thinker_->get_input_embeddings(input_ids, input_params);
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    return thinker_->get_multimodal_embeddings(input_params);
  }
  layer::NpuLmHead get_npu_lm_head() { return thinker_->get_npu_lm_head(); }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    thinker_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return thinker_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    thinker_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3OmniMoe_Thinker_ForConditionalGeneration thinker_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_ForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen3_omni_moe, Qwen3OmniMoe_InputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe,
                          Qwen3OmniMoe_ForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_omni_moe, Qwen2VLImageProcessor);
REGISTER_FEATURE_EXTRACTOR(qwen3_omni_moe, WhisperFeatureExtractor);

REGISTER_MODEL_ARGS(qwen3_omni_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe");

  // feature extractor default config
  LOAD_ARG_OR(has_feature_extractor, "has_feature_extractor", true);
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  // padding_strategy default to be PADDING_STRATEGT::LONGEST
  LOAD_ARG_OR(mm_audio_padding_strategy, "padding_strategy", 1);
  LOAD_ARG_OR(mm_audio_max_length, "max_length", -1);
  LOAD_ARG_OR(mm_audio_pad_to_multiple_of, "pad_to_multiple_of", -1);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);
  LOAD_ARG_OR(
      mm_audio_return_token_timestamps, "return_token_timestamps", false);
  LOAD_ARG_OR(mm_audio_return_attention_mask, "return_attention_mask", true);
  // TODO: use gflag instead
  LOAD_ARG_OR(
      mm_use_audio_in_video, "use_audio_in_video", FLAGS_use_audio_in_video);
  LOAD_ARG_OR(mm_position_id_per_seconds, "position_id_per_seconds", 13);
  LOAD_ARG_OR(mm_fps, "fps", 1.0);
  LOAD_ARG_OR(mm_temporal_patch_size, "temporal_patch_size", 2);

  // thinker config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config", [&] {
    LOAD_ARG_OR_PREFIX(vision_start_token_id, "vision_start_token_id", 151652);
    LOAD_ARG_OR_PREFIX(vision_end_token_id, "vision_end_token_id", 151653);
    LOAD_ARG_OR_PREFIX(vision_token_id, "vision_token_id", 151654);
    LOAD_ARG_OR_PREFIX(image_token_id, "image_token_id", 151655);
    LOAD_ARG_OR_PREFIX(video_token_id, "video_token_id", 151656);
    LOAD_ARG_OR_PREFIX(audio_token_id, "audio_token_id", 151675);
    LOAD_ARG_OR_PREFIX(audio_start_token_id, "audio_start_token_id", 151669);
    LOAD_ARG_OR_PREFIX(audio_end_token_id, "audio_end_token_id", 151670);
    LOAD_ARG_OR_PREFIX(dtype, "dtype", "bfloat16");
  });

  // thinker.text_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.text_config", [&] {
    LOAD_ARG_OR_PREFIX(attention_bias, "attention_bias", false);
    LOAD_ARG_OR_PREFIX(attention_dropout, "attention_dropout", 0.0);
    LOAD_ARG_OR_PREFIX(decoder_sparse_step, "decoder_sparse_step", 1);

    LOAD_ARG_OR_PREFIX(bos_token_id, "bos_token_id", 151643);
    LOAD_ARG_OR_PREFIX(eos_token_id, "eos_token_id", 151645);
    LOAD_ARG_OR_PREFIX(hidden_act, "hidden_act", "silu");
    LOAD_ARG_OR_PREFIX(hidden_size, "hidden_size", 2048);
    LOAD_ARG_OR_PREFIX(intermediate_size, "intermediate_size", 768);
    LOAD_ARG_OR_PREFIX(
        max_position_embeddings, "max_position_embeddings", 65536);
    LOAD_ARG_OR_PREFIX(max_window_layers, "max_window_layers", 28);
    LOAD_ARG_OR_PREFIX(n_heads, "num_attention_heads", 32);
    LOAD_ARG_OR_PREFIX(n_layers, "num_hidden_layers", 48);
    LOAD_ARG_OR_PREFIX(n_kv_heads, "num_key_value_heads", 4);
    LOAD_ARG_OR_PREFIX(rms_norm_eps, "rms_norm_eps", 1e-06);
    LOAD_ARG_OR_PREFIX(sliding_window, "sliding_window", 32768);
    LOAD_ARG_OR_PREFIX(tie_word_embeddings, "tie_word_embeddings", false);
    LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
    LOAD_ARG_OR_PREFIX(initializer_range, "initializer_range", 0.02);
    LOAD_ARG_OR_PREFIX(use_sliding_window, "use_sliding_window", false);
    LOAD_ARG_OR_PREFIX(moe_intermediate_size, "moe_intermediate_size", 768);
    LOAD_ARG_OR_PREFIX(norm_topk_prob, "norm_topk_prob", true);
    LOAD_ARG_OR_PREFIX(num_experts, "num_experts", 128);
    LOAD_ARG_OR_PREFIX(num_experts_per_tok, "num_experts_per_tok", 8);
    LOAD_ARG_OR_FUNC_PREFIX(head_dim, "head_dim", [&] {
      return args->hidden_size() / args->n_heads();
    });
    LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
    LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
    LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());
    SET_ARG(stop_token_ids,
            std::unordered_set<int32_t>({args->eos_token_id()}));
    LOAD_ARG_OR_PREFIX(rope_scaling_rope_type, "rope_scaling.type", "mrope");
    LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
    LOAD_ARG_OR_PREFIX(rope_theta, "rope_theta", 1000000.0f);
    LOAD_ARG_OR_PREFIX(vocab_size, "vocab_size", 152064);
  });

  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }

  // thinker.vision_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.vision_config", [&] {
    LOAD_ARG_OR_PREFIX(mm_num_hidden_layers, "depth", 27);
    LOAD_ARG_OR_PREFIX(mm_hidden_act, "hidden_act", "gelu_pytorch_tanh");
    LOAD_ARG_OR_PREFIX(mm_hidden_size, "hidden_size", 1152);
    LOAD_ARG_OR_PREFIX(mm_intermediate_size, "intermediate_size", 4304);
    LOAD_ARG_OR_PREFIX(mm_num_attention_heads, "num_heads", 16);
    LOAD_ARG_OR_PREFIX(mm_num_channels, "in_channels", 3);
    LOAD_ARG_OR_PREFIX(mm_projection_dim, "out_hidden_size", 2048);
    LOAD_ARG_OR_PREFIX(mm_patch_size, "patch_size", 16);
    LOAD_ARG_OR_PREFIX(
        mm_num_position_embeddings, "num_position_embeddings", 2304);
    LOAD_ARG_OR_PREFIX(mm_spatial_merge_size, "spatial_merge_size", 2);
    LOAD_ARG_PREFIX(mm_deepstack_visual_indexes, "deepstack_visual_indexes");
    LOAD_ARG_OR_PREFIX(mm_temporal_patch_size, "temporal_patch_size", 2);
    LOAD_ARG_OR_PREFIX(mm_image_size, "image_size", 768);

    LOAD_ARG_OR_FUNC_PREFIX(mm_head_dim, "head_dim", [&] {
      return args->mm_hidden_size() / args->mm_num_attention_heads();
    });
  });

  // thinker.audio_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.audio_config", [&] {
    LOAD_ARG_OR_PREFIX(
        mm_audio_num_attention_heads, "encoder_attention_heads", 20);
    LOAD_ARG_OR_PREFIX(mm_audio_hidden_size, "d_model", 1280);
    LOAD_ARG_OR_PREFIX(
        mm_audio_downsample_hidden_size, "downsample_hidden_size", 480);
    LOAD_ARG_OR_PREFIX(mm_audio_d_model, "d_model", 1280);
    LOAD_ARG_OR_PREFIX(mm_audio_num_mel_bins, "num_mel_bins", 128);
    LOAD_ARG_OR_PREFIX(
        mm_audio_max_source_positions, "max_source_positions", 1500);
    LOAD_ARG_OR_PREFIX(mm_audio_scale_embedding, "scale_embedding", false);
    LOAD_ARG_OR_PREFIX(mm_audio_n_window, "n_window", 50);
    LOAD_ARG_OR_PREFIX(mm_audio_n_window_infer, "n_window_infer", 800);
    LOAD_ARG_OR_PREFIX(mm_audio_conv_chunksize, "conv_chunksize", 500);
    LOAD_ARG_OR_PREFIX(mm_audio_encoder_layers, "encoder_layers", 32);
    LOAD_ARG_OR_PREFIX(mm_audio_output_dim, "output_dim", 2048);
  });
});

}  // namespace xllm
