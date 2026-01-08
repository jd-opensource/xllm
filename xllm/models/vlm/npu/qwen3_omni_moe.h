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

class Qwen3OmniMoe_InputProcessor : public InputProcessor {
  enum class TokenType { INVALID, IMAGE, VIDEO, AUDIO };

  // A pointer reference class that contains the token of current modality,
  // index and the expected embedding length of the current modality
  struct ModalityIndexRef {
    uint32_t* modality_index_ptr = 0;
    const std::string* modality_token_ptr = nullptr;
    // audio : [feat_length]
    // image or video : [grid_thw]
    const torch::Tensor* modality_size_ptr = nullptr;
    uint32_t modality_nums_ = 0;

    void safe_check_modality() {
      CHECK(*modality_index_ptr < modality_nums_)
          << "The index of " << *modality_token_ptr
          << " modality is out of range, have " << modality_nums_
          << " modality inputs "
          << "but try to access index " << *modality_index_ptr;
    }

    ModalityIndexRef() = default;

    ModalityIndexRef(uint32_t* modality_index_ptr,
                     const std::string* modality_token_ptr,
                     const torch::Tensor* modality_size_ptr)
        : modality_index_ptr(modality_index_ptr),
          modality_token_ptr(modality_token_ptr),
          modality_size_ptr(modality_size_ptr) {
      if (modality_size_ptr->defined()) {
        modality_nums_ = modality_size_ptr->size(0);
      }
    }
  };

 public:
  Qwen3OmniMoe_InputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
    fps_ = args.mm_fps();
    temporal_patch_size_ = args.mm_temporal_patch_size();
    use_audio_in_video_ = args.mm_use_audio_in_video();
    video_second_per_grid_ = temporal_patch_size_ / fps_;
    vision_start_token_id_ = args.vision_start_token_id();
    vision_end_token_id_ = args.vision_end_token_id();
    image_token_id_ = args.image_token_id();
    video_token_id_ = args.video_token_id();
    audio_token_id_ = args.audio_token_id();
    audio_start_token_id_ = args.audio_start_token_id();
    audio_end_token_id_ = args.audio_end_token_id();
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor feat_length;
    if (auto res = mm_data.get<torch::Tensor>("feat_length"))
      feat_length = res.value();

    if (!image_grid_thw.defined() && !video_grid_thw.defined() &&
        !feat_length.defined())
      return;

    auto merge_length = merge_size_ * merge_size_;

    uint32_t total_audio_token = 0;
    if (feat_length.defined()) {
      auto count = feat_length.sizes()[0];
      for (size_t idx = 0; idx < count; ++idx)
        total_audio_token += feat_length[idx].item<int>();
    }

    uint32_t total_image_token = 0;
    if (image_grid_thw.defined()) {
      auto count = image_grid_thw.sizes()[0];
      for (size_t idx = 0; idx < count; ++idx)
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
    }

    uint32_t total_video_token = 0;
    if (video_grid_thw.defined()) {
      auto count = video_grid_thw.sizes()[0];
      for (size_t idx = 0; idx < count; ++idx) {
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
      }
    }

    uint32_t total_token_len = total_image_token * image_token_.size() +
                               total_video_token * video_token_.size() +
                               total_audio_token * audio_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);

    uint32_t image_index = 0;
    uint32_t video_index = 0;
    uint32_t audio_index = 0;

    uint32_t* modality_index_ptr = 0;
    const std::string* modality_token_ptr = nullptr;
    const torch::Tensor* modality_size_ptr = nullptr;
    std::unordered_map<TokenType, ModalityIndexRef> modality_index_map = {
        {TokenType::AUDIO,
         ModalityIndexRef(&audio_index, &audio_token_, &feat_length)},
        {TokenType::IMAGE,
         ModalityIndexRef(&image_index, &image_token_, &image_grid_thw)},
        {TokenType::VIDEO,
         ModalityIndexRef(&video_index, &video_token_, &video_grid_thw)}};

    size_t begin = 0;
    auto pair = find_special_token(prompt, begin);
    FLAGS_max_log_size = 1000000;
    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);

      auto cur_modality = modality_index_map[pair.first];
      cur_modality.safe_check_modality();
      modality_size_ptr = cur_modality.modality_size_ptr;
      modality_token_ptr = cur_modality.modality_token_ptr;
      modality_index_ptr = cur_modality.modality_index_ptr;
      if (pair.first == TokenType::AUDIO) {
        // for audio
        auto token_num =
            (*modality_size_ptr)[(*modality_index_ptr)].item<int32_t>();
        while (token_num--) data.append(*modality_token_ptr);
      } else if (pair.first == TokenType::VIDEO && use_audio_in_video_) {
        // for audio in video
        auto audio_modality = modality_index_map[TokenType::AUDIO];
        audio_modality.safe_check_modality();
        auto audio_size_ptr = audio_modality.modality_size_ptr;
        auto audio_token_ptr = audio_modality.modality_token_ptr;
        auto audio_index_ptr = audio_modality.modality_index_ptr;

        auto audio_token_indices =
            torch::arange((*audio_size_ptr)[(*audio_index_ptr)].item<int32_t>())
                .to(torch::kInt32);
        auto video_grid_thw = (*modality_size_ptr)[(*modality_index_ptr)];
        int32_t T = video_grid_thw[0].item<int32_t>();
        int32_t H = video_grid_thw[1].item<int32_t>();
        int32_t W = video_grid_thw[2].item<int32_t>();

        int32_t height = H / merge_size_;
        int32_t width = W / merge_size_;

        auto video_token_indices_1d = torch::arange(T);
        auto video_token_indices = video_token_indices_1d.view({T, 1, 1});

        video_token_indices = video_token_indices.expand({T, height, width});

        video_token_indices = video_token_indices.reshape({-1});
        video_token_indices = video_token_indices * video_second_per_grid_ *
                              position_id_per_seconds_;
        auto video_indices_vec = video_token_indices.accessor<float, 1>();
        auto audio_indices_vec = audio_token_indices.accessor<int32_t, 1>();

        std::string placeholder_string = audio_bos_token_;

        size_t video_data_index = 0;
        size_t audio_data_index = 0;
        size_t video_len = video_indices_vec.size(0);
        size_t audio_len = audio_indices_vec.size(0);
        bool indexer = true;
        while (video_data_index < video_len && audio_data_index < audio_len) {
          if (video_indices_vec[video_data_index] <=
              audio_indices_vec[audio_data_index]) {
            placeholder_string.append(*modality_token_ptr);
            video_data_index++;
          } else {
            placeholder_string.append(*audio_token_ptr);
            audio_data_index++;
          }
        }

        if (video_data_index < video_len) {
          size_t remaining_video = video_len - video_data_index;
          for (size_t i = 0; i < remaining_video; ++i) {
            placeholder_string.append(*modality_token_ptr);
          }
        }

        if (audio_data_index < audio_len) {
          size_t remaining_audio = audio_len - audio_data_index;
          for (size_t i = 0; i < remaining_audio; ++i) {
            placeholder_string.append(*audio_token_ptr);
          }
        }

        placeholder_string.append(audio_eos_token_);
        data.append(placeholder_string);
        ++(*audio_index_ptr);
      } else {
        // for image and video
        auto token_num =
            (*modality_size_ptr)[(*modality_index_ptr)].prod().item<int32_t>() /
            merge_length;
        while (token_num--) data.append(*modality_token_ptr);
      }
      ++(*modality_index_ptr);
      begin = pair.second + modality_token_ptr->size();
      pair = find_special_token(prompt, begin);
    }

    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);

    prompt = std::move(data);
  }

  void find_mm_spans(const std::vector<int>& prompt, MMData& mm_data) {
    auto start = prompt.begin();
    uint32_t global_mm_index = 0;
    uint32_t offset = 0;
    uint32_t length = 0;
    auto& mm_items = mm_data.items<MMItemVec>();
    while (true) {
      auto vision_start_it =
          std::find(start, prompt.end(), vision_start_token_id_);
      auto vision_end_it = std::find(start, prompt.end(), vision_end_token_id_);
      auto audio_start_it =
          std::find(start, prompt.end(), audio_start_token_id_);
      auto audio_end_it = std::find(start, prompt.end(), audio_end_token_id_);
      // vision_start_it == audio_start_it when reach the end
      if (vision_start_it == audio_start_it) {
        break;
      }

      auto min_start_it = std::min(vision_start_it, audio_start_it);
      auto min_end_it = std::min(vision_end_it, audio_end_it);
      auto max_end_it = std::max(vision_end_it, audio_end_it);
      offset = std::distance(prompt.begin(), min_start_it);
      length = std::distance(min_start_it + 1, min_end_it);
      auto& item = mm_items[global_mm_index];

      // use_audio_in_video case, offset subtract the audio_start_token,
      // length subtract the audio_start_token and audio_end_token
      if (*min_start_it == vision_start_token_id_ &&
          (*(min_start_it + 1) == audio_start_token_id_)) {
        item.mutable_state().mutable_token_pos() = {offset + 2, length - 1};
        std::vector<int> audio_in_video_propmt(
            prompt.begin() + offset + 2,
            prompt.begin() + offset + 2 + length - 1);
        torch::Tensor audio_in_video_mask =
            torch::tensor(audio_in_video_propmt, torch::kInt32);
        item.add("audio_in_video_mask", audio_in_video_mask);
        // audio_in_video case, vision end is always greater than audio_end
        min_end_it = max_end_it;
      } else {
        item.mutable_state().mutable_token_pos() = {offset + 1, length};
      }
      global_mm_index++;
      start = std::next(min_end_it);
    }
  }

 private:
  std::pair<TokenType, size_t> find_special_token(const std::string& prompt,
                                                  size_t begin) {
    struct TokenInfo {
      const std::string& token;
      TokenType type;
      size_t pos;
    };

    std::vector<TokenInfo> tokens = {
        {image_token_, TokenType::IMAGE, std::string::npos},
        {video_token_, TokenType::VIDEO, std::string::npos},
        {audio_token_, TokenType::AUDIO, std::string::npos}};

    for (auto& token_info : tokens) {
      token_info.pos = prompt.find(token_info.token, begin);
    }

    auto earliest =
        std::min_element(tokens.begin(),
                         tokens.end(),
                         [](const TokenInfo& a, const TokenInfo& b) {
                           if (a.pos == std::string::npos) return false;
                           if (b.pos == std::string::npos) return true;
                           return a.pos < b.pos;
                         });

    if (earliest == tokens.end() || earliest->pos == std::string::npos) {
      return {TokenType::INVALID, std::string::npos};
    }

    return {earliest->type, earliest->pos};
  }

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  const std::string audio_token_ = "<|audio_pad|>";
  const std::string vision_bos_token_ = "<|vision_start|>";
  const std::string vision_end_token_ = "<|vision_end|>";
  const std::string audio_bos_token_ = "<|audio_start|>";
  const std::string audio_eos_token_ = "<|audio_end|>";

  int32_t vision_start_token_id_;
  int32_t vision_end_token_id_;
  int32_t image_token_id_;
  int32_t video_token_id_;
  int32_t audio_token_id_;
  int32_t audio_start_token_id_;
  int32_t audio_end_token_id_;

  int32_t merge_size_ = 0;
  int32_t position_id_per_seconds_ = 13;
  double fps_ = 1.0;
  int32_t temporal_patch_size_ = 2.0;
  bool use_audio_in_video_ = false;
  double video_second_per_grid_ = 0.0;
};

class Qwen3OmniMoe_ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    thinker_ = register_module(
        "thinker", Qwen3OmniMoe_Thinker_ForConditionalGeneration(context));
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    auto emb = thinker_(tokens, positions, kv_caches, input_params);

    return emb;
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
    LOAD_ARG_OR_PREFIX(rope_scaling_rope_type, "rope_scaling.type", "mrope");
    LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
    LOAD_ARG_OR_PREFIX(rope_theta, "rope_theta", 1000000.0f);
    LOAD_ARG_OR_PREFIX(vocab_size, "vocab_size", 152064);
  });

  if (args->rope_scaling_rope_type() == "default")
    args->rope_scaling_rope_type() = "mrope";

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
