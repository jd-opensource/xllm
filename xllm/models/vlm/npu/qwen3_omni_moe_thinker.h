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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <unistd.h>

#include <unordered_map>

#include "core/common/global_flags.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_qwen3_audio_encoder_layer_impl.h"
#include "core/layers/npu/npu_qwen3_vision_encoder_layer_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"
#include "processors/feature_extraction_whisper.h"
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "qwen2_5_vl.h"
#include "qwen3_vl.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm {

class Qwen3OmniMoe_InputProcessor : public InputProcessor {
  enum class TokenType { INVALID, IMAGE, VIDEO, AUDIO };

  // A pointer reference class that contains the token of current modality,
  // index and the expected embedding length of the current modality
  struct ModalityIndexRef {
    uint32_t modality_index = 0;
    const std::string modality_token;
    // audio : [feat_length]
    // image or video : [grid_thw]
    const torch::Tensor* modality_size_ptr = nullptr;
    uint32_t modality_nums_ = 0;

    torch::Tensor get_modality_size() {
      CHECK(modality_index < modality_nums_)
          << "The index of " << modality_token
          << " modality is out of range, have " << modality_nums_
          << " modality inputs "
          << "but try to access index " << modality_index;
      return (*modality_size_ptr)[modality_index++];
    }

    const std::string& get_modality_token() { return modality_token; }

    ModalityIndexRef() = default;

    ModalityIndexRef(uint32_t& modality_index,
                     const std::string& modality_token,
                     const torch::Tensor* modality_size_ptr)
        : modality_index(modality_index),
          modality_token(modality_token),
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
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    torch::Tensor feat_length;
    if (auto res = mm_data.get<torch::Tensor>("feat_length")) {
      feat_length = res.value();
    }

    if (!image_grid_thw.defined() && !video_grid_thw.defined() &&
        !feat_length.defined()) {
      return;
    }

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
    std::unordered_map<TokenType, ModalityIndexRef> modality_index_map = {};
    modality_index_map.emplace(
        TokenType::AUDIO,
        ModalityIndexRef(audio_index, audio_token_, &feat_length)),
        modality_index_map.emplace(
            TokenType::IMAGE,
            ModalityIndexRef(image_index, image_token_, &image_grid_thw)),
        modality_index_map.emplace(
            TokenType::VIDEO,
            ModalityIndexRef(video_index, video_token_, &video_grid_thw));

    size_t begin = 0;
    auto pair = find_special_token(prompt, begin);
    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);

      auto& cur_modality = modality_index_map[pair.first];
      auto modality_size_ref = cur_modality.get_modality_size();
      auto modality_token_ref = cur_modality.get_modality_token();
      if (pair.first == TokenType::AUDIO) {
        // for audio
        auto token_num = modality_size_ref.item<int32_t>();
        while (token_num--) data.append(modality_token_ref);
      } else if (pair.first == TokenType::VIDEO && use_audio_in_video_) {
        // for audio in video
        auto& audio_modality = modality_index_map[TokenType::AUDIO];
        auto audio_size_ref = audio_modality.get_modality_size();
        auto audio_token_ref = audio_modality.get_modality_token();
        auto audio_token_indices =
            torch::arange(audio_size_ref.item<int32_t>()).to(torch::kInt32);

        int32_t T = modality_size_ref[0].item<int32_t>();
        int32_t H = modality_size_ref[1].item<int32_t>();
        int32_t W = modality_size_ref[2].item<int32_t>();

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
            placeholder_string.append(modality_token_ref);
            video_data_index++;
          } else {
            placeholder_string.append(audio_token_ref);
            audio_data_index++;
          }
        }

        if (video_data_index < video_len) {
          size_t remaining_video = video_len - video_data_index;
          for (size_t i = 0; i < remaining_video; ++i) {
            placeholder_string.append(modality_token_ref);
          }
        }

        if (audio_data_index < audio_len) {
          size_t remaining_audio = audio_len - audio_data_index;
          for (size_t i = 0; i < remaining_audio; ++i) {
            placeholder_string.append(audio_token_ref);
          }
        }

        placeholder_string.append(audio_eos_token_);
        data.append(placeholder_string);
      } else {
        // for image and video
        auto token_num =
            modality_size_ref.prod().item<int32_t>() / merge_length;
        while (token_num--) data.append(modality_token_ref);
      }
      begin = pair.second + modality_token_ref.size();
      pair = find_special_token(prompt, begin);
    }

    if (begin < prompt.size()) {
      data.append(prompt, begin, std::string::npos);
    }

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
                           if (a.pos == std::string::npos) {
                             return false;
                           }
                           if (b.pos == std::string::npos) {
                             return true;
                           }
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

class SinusoidsPositionEmbeddingImpl : public torch::nn::Module {
 public:
  SinusoidsPositionEmbeddingImpl(int64_t length,
                                 int64_t channels,
                                 double max_timescale = 10000.0) {
    if (channels % 2 != 0) {
      CHECK(false) << "SinusoidsPositionEmbedding needs even channels input";
    }

    double log_timescale_increment =
        std::log(max_timescale) / (channels / 2 - 1);
    torch::Tensor inv_timescales =
        torch::exp(-log_timescale_increment * torch::arange(channels / 2))
            .to(torch::kFloat32);

    torch::Tensor scaled_time =
        torch::arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0);

    pos_embedding_ =
        torch::cat({torch::sin(scaled_time), torch::cos(scaled_time)}, 1);
  }

  torch::Tensor forward(int64_t seqlen) {
    return pos_embedding_.slice(0, 0, seqlen);
  }

 private:
  torch::Tensor pos_embedding_;
};

TORCH_MODULE(SinusoidsPositionEmbedding);

class Qwen3OmniMoe_Thinker_AudioBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_AudioBlockImpl(const ModelContext& context) {
    // register submodules
    encoder_layer_ = register_module("encoder_layer",
                                     layer::NpuQwen3AudioEncoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cu_seq_len,
                        std::vector<int>& cu_seq_len_vec,
                        ModelInputParams& input_params,
                        int node_id) {
    return encoder_layer_(x, cu_seq_len, cu_seq_len_vec, input_params, node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuQwen3AudioEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_AudioBlock);

class Qwen3OmniMoe_Thinker_AudioTransformerImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_AudioTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    options_ = context.get_tensor_options();
    auto downsample_hidden_size = model_args.mm_audio_downsample_hidden_size();
    embed_dim_ = model_args.mm_audio_d_model();
    num_mel_bins_ = model_args.mm_audio_num_mel_bins();
    max_source_positions_ = model_args.mm_audio_max_source_positions();
    embed_scale_ =
        model_args.mm_audio_scale_embedding() ? std::sqrt(embed_dim_) : 1.0;
    n_window_ = model_args.mm_audio_n_window();

    positional_embedding_ = register_module(
        "positional_embedding",
        SinusoidsPositionEmbedding(max_source_positions_, embed_dim_));

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < model_args.mm_audio_encoder_layers(); ++i) {
      auto layer = Qwen3OmniMoe_Thinker_AudioBlock(context);
      layers_->push_back(layer);
    }

    ln_post_ = register_module(
        "ln_post",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})
                                 .elementwise_affine(true)));

    conv2d1_ = register_module(
        "conv2d1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));

    conv2d2_ = register_module(
        "conv2d2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              downsample_hidden_size, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));

    conv2d3_ = register_module(
        "conv2d3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              downsample_hidden_size, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));

    int64_t conv_output_dim = ((((num_mel_bins_ + 1) / 2 + 1) / 2 + 1) / 2);

    conv_out_ = register_module(
        "conv_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(downsample_hidden_size * conv_output_dim,
                                     embed_dim_)
                .bias(false)));

    proj1_ = register_module(
        "proj1",
        torch::nn::Linear(
            torch::nn::LinearOptions(embed_dim_, embed_dim_).bias(true)));

    proj2_ = register_module(
        "proj2",
        torch::nn::Linear(torch::nn::LinearOptions(
                              embed_dim_, model_args.mm_audio_output_dim())
                              .bias(true)));

    n_window_infer_ = model_args.mm_audio_n_window_infer();
    conv_chunksize_ = model_args.mm_audio_conv_chunksize();
  }

  torch::Tensor forward(const torch::Tensor& input_features,
                        const ModelInputParams& input_params,
                        const torch::Tensor& feature_lens = torch::Tensor()) {
    auto aftercnn_lens_calc =
        audio_utils::get_feat_extract_output_lengths(feature_lens);

    auto chunk_num =
        torch::ceil(feature_lens / (n_window_ * 2)).to(torch::kLong);
    int64_t total_chunks = chunk_num.sum().item<int64_t>();

    auto chunk_lengths = torch::full({total_chunks},
                                     n_window_ * 2,
                                     torch::TensorOptions()
                                         .dtype(torch::kLong)
                                         .device(feature_lens.device()));

    auto padded_chunk_num = torch::nn::functional::pad(
        chunk_num, torch::nn::functional::PadFuncOptions({1, 0}).value(-1));
    auto tail_chunk_index = padded_chunk_num.cumsum(0).slice(0, 1);

    auto remainder = feature_lens % (n_window_ * 2);
    chunk_lengths.index_put_({torch::indexing::TensorIndex(tail_chunk_index)},
                             remainder);
    chunk_lengths.index_put_({torch::indexing::TensorIndex(chunk_lengths == 0)},
                             n_window_ * 2);

    auto input_t = input_features.t();
    auto chunk_lengths_list = chunk_lengths.to(torch::kCPU).contiguous();

    auto chunk_lengths_cpu = chunk_lengths.to(torch::kCPU).contiguous();
    at::IntArrayRef split_sizes(chunk_lengths_cpu.data_ptr<int64_t>(),
                                static_cast<size_t>(chunk_lengths_cpu.size(0)));

    auto chunk_list = input_t.split_with_sizes(split_sizes, 0);

    auto padded_feature =
        torch::nn::utils::rnn::pad_sequence(chunk_list, true).transpose(1, 2);
    auto feature_lens_after_cnn = audio_utils::get_feat_extract_output_lengths(
        chunk_lengths.to(torch::kLong));

    std::vector<torch::Tensor> mask_tensors;
    for (int64_t i = 0; i < feature_lens_after_cnn.size(0); ++i) {
      int64_t length = feature_lens_after_cnn[i].item<int64_t>();
      mask_tensors.push_back(torch::full({length},
                                         1,
                                         torch::TensorOptions()
                                             .dtype(torch::kBool)
                                             .device(padded_feature.device())));
    }

    auto padded_mask_after_cnn =
        torch::nn::utils::rnn::pad_sequence(mask_tensors, true);

    padded_feature = padded_feature.unsqueeze(1);

    std::vector<torch::Tensor> padded_embeds;
    int64_t batch_size = padded_feature.size(0);
    for (int64_t start = 0; start < batch_size; start += conv_chunksize_) {
      int64_t end = std::min(start + conv_chunksize_, batch_size);
      auto chunk = padded_feature.slice(0, start, end);
      auto embed = torch::gelu(conv2d1_(chunk));
      embed = torch::gelu(conv2d2_(embed));
      embed = torch::gelu(conv2d3_(embed));
      padded_embeds.push_back(embed);
    }

    auto padded_embed = torch::cat(padded_embeds, 0);
    auto [b, c, f, t] = std::make_tuple(padded_embed.size(0),
                                        padded_embed.size(1),
                                        padded_embed.size(2),
                                        padded_embed.size(3));

    auto reshaped =
        padded_embed.permute({0, 3, 1, 2}).contiguous().view({b, t, c * f});
    padded_embed = conv_out_(reshaped);
    auto pos_embed = positional_embedding_->forward(padded_embed.size(1))
                         .unsqueeze(0)
                         .to(options_.device(), padded_embed.dtype());

    padded_embed = padded_embed + pos_embed;

    auto hidden_states =
        padded_embed
            .masked_select(
                padded_mask_after_cnn.unsqueeze(-1).expand_as(padded_embed))
            .view({-1, padded_embed.size(-1)});
    auto window_aftercnn =
        padded_mask_after_cnn.size(-1) * (n_window_infer_ / (n_window_ * 2));

    std::vector<int> cu_chunk_lens = {};
    for (int64_t i = 0; i < aftercnn_lens_calc.size(0); ++i) {
      int64_t cnn_len = static_cast<int>(aftercnn_lens_calc[i].item<long>());
      int64_t full_windows = cnn_len / window_aftercnn;
      for (int64_t j = 0; j < full_windows; ++j) {
        cu_chunk_lens.push_back(window_aftercnn);
      }
      int64_t remainder = cnn_len % window_aftercnn;
      if (remainder != 0) {
        cu_chunk_lens.push_back(remainder);
      }
    }

    auto cu_seqlens = torch::tensor(cu_chunk_lens,
                                    torch::TensorOptions()
                                        .device(aftercnn_lens_calc.device())
                                        .dtype(torch::kInt32))
                          .to(torch::kInt32);
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int>(),  // full seqlen vec
        cu_seqlens_cpu.data_ptr<int>() + cu_seqlens_cpu.numel());

    for (int idx = 0; idx < layers_->size(); ++idx) {
      hidden_states =
          layers_[idx]->as<Qwen3OmniMoe_Thinker_AudioBlock>()->forward(
              hidden_states, cu_seqlens, cu_seqlens_vec, input_params_new, idx);
    }

    hidden_states = ln_post_(hidden_states);
    hidden_states = proj1_(hidden_states);
    hidden_states = torch::gelu(hidden_states);
    hidden_states = proj2_(hidden_states);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "proj1.weight", proj1_->weight, is_proj1_weight_loaded_);
    weight::load_weight(
        state_dict, "proj1.bias", proj1_->bias, is_proj1_bias_loaded_);
    weight::load_weight(
        state_dict, "proj2.weight", proj2_->weight, is_proj2_weight_loaded_);
    weight::load_weight(
        state_dict, "proj2.bias", proj2_->bias, is_proj2_bias_loaded_);

    weight::load_weight(state_dict,
                        "conv2d1.weight",
                        conv2d1_->weight,
                        is_conv2d1_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d1.bias", conv2d1_->bias, is_conv2d1_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv2d2.weight",
                        conv2d2_->weight,
                        is_conv2d2_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d2.bias", conv2d2_->bias, is_conv2d2_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv2d3.weight",
                        conv2d3_->weight,
                        is_conv2d3_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d3.bias", conv2d3_->bias, is_conv2d3_bias_loaded_);

    weight::load_weight(state_dict,
                        "ln_post.weight",
                        ln_post_->weight,
                        is_ln_post_weight_loaded_);
    weight::load_weight(
        state_dict, "ln_post.bias", ln_post_->bias, is_ln_post_bias_loaded_);
    for (size_t idx = 0; idx < layers_->size(); idx++) {
      auto prefix = "layers." + std::to_string(idx) + ".";
      layers_[idx]->as<Qwen3OmniMoe_Thinker_AudioBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << "conv_out.weight";
    CHECK(is_proj1_weight_loaded_)
        << "weight is not loaded for " << "proj1.weight";
    CHECK(is_proj1_bias_loaded_) << "weight is not loaded for " << "proj1.bias";
    CHECK(is_proj2_weight_loaded_)
        << "weight is not loaded for " << "proj2.weight";
    CHECK(is_proj2_bias_loaded_) << "weight is not loaded for " << "proj2.bias";

    CHECK(is_conv2d1_weight_loaded_)
        << "weight is not loaded for " << "conv2d1.weight";
    CHECK(is_conv2d1_bias_loaded_)
        << "weight is not loaded for " << "conv2d1.bias";
    CHECK(is_conv2d2_weight_loaded_)
        << "weight is not loaded for " << "conv2d2.weight";
    CHECK(is_conv2d2_bias_loaded_)
        << "weight is not loaded for " << "conv2d2.bias";
    CHECK(is_conv2d3_weight_loaded_)
        << "weight is not loaded for " << "conv2d3.weight";
    CHECK(is_conv2d3_bias_loaded_)
        << "weight is not loaded for " << "conv2d3.bias";

    CHECK(is_ln_post_weight_loaded_)
        << "weight is not loaded for " << "ln_post.weight";
    CHECK(is_ln_post_bias_loaded_)
        << "weight is not loaded for " << "ln_post.bias";
    for (int idx = 0; idx < layers_->size(); ++idx) {
      auto prefix = "layers." + std::to_string(idx) + ".";
      layers_[idx]
          ->as<Qwen3OmniMoe_Thinker_AudioBlock>()
          ->verify_loaded_weights(prefix);
    }
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < layers_->size(); ++idx) {
      layers_[idx]
          ->as<Qwen3OmniMoe_Thinker_AudioBlock>()
          ->merge_loaded_weights();
    }
  }

 private:
  int64_t embed_dim_;
  int64_t num_mel_bins_;
  int64_t max_source_positions_;
  double embed_scale_;
  int64_t n_window_;
  int64_t n_window_infer_;
  int64_t conv_chunksize_;
  torch::TensorOptions options_;

  SinusoidsPositionEmbedding positional_embedding_{nullptr};
  torch::nn::ModuleList layers_{nullptr};
  torch::nn::LayerNorm ln_post_{nullptr};

  torch::nn::Conv2d conv2d1_{nullptr};
  torch::nn::Conv2d conv2d2_{nullptr};
  torch::nn::Conv2d conv2d3_{nullptr};

  torch::nn::Linear conv_out_{nullptr};
  torch::nn::Linear proj1_{nullptr};
  torch::nn::Functional act_{nullptr};
  torch::nn::Linear proj2_{nullptr};

  bool is_conv_out_weight_loaded_ = false;
  bool is_conv2d1_weight_loaded_ = false;
  bool is_conv2d1_bias_loaded_ = false;
  bool is_conv2d2_weight_loaded_ = false;
  bool is_conv2d2_bias_loaded_ = false;
  bool is_conv2d3_weight_loaded_ = false;
  bool is_conv2d3_bias_loaded_ = false;
  bool is_ln_post_weight_loaded_ = false;
  bool is_ln_post_bias_loaded_ = false;
  bool is_proj1_weight_loaded_ = false;
  bool is_proj1_bias_loaded_ = false;
  bool is_proj2_weight_loaded_ = false;
  bool is_proj2_bias_loaded_ = false;
};

TORCH_MODULE(Qwen3OmniMoe_Thinker_AudioTransformer);

class Qwen3OmniMoe_Thinker_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_temporal_patch_size() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();

    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(true)));

    proj_->weight.set_data(proj_->weight.to(options));
    proj_->bias.set_data(proj_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor x) { return proj_(x); }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }
    auto bias = state_dict.get_tensor("proj.bias");
    if (bias.defined()) {
      bias = bias.reshape({bias.size(0)});
      DCHECK_EQ(proj_->bias.sizes(), bias.sizes())
          << "proj bias size mismatch for " << name();
      proj_->bias.data().copy_(bias);
      proj_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(proj_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  bool proj_weight_loaded_ = false;
  bool proj_bias_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_VisionPatchEmbed);

class Qwen3OmniMoe_Thinker_VisionBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_VisionBlockImpl(const ModelContext& context) {
    // register submodules
    encoder_layer_ = register_module(
        "encoder_layer", layer::NpuQwen3VisionEncoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int>& cu_seq_len_vec,
                        ModelInputParams& input_params,
                        int node_id) {
    return encoder_layer_(x,
                          m_cos_pos,
                          m_sin_pos,
                          cu_seq_len,
                          cu_seq_len_vec,
                          input_params,
                          node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuQwen3VisionEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_VisionBlock);

class Qwen3OmniMoe_Thinker_VisionRotaryEmbeddingImpl
    : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_VisionRotaryEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    dim_ = model_args.mm_head_dim() / 2;
    theta_ = 10000.0;

    auto opts = options.dtype(torch::kFloat32);
    auto inv_freq =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, opts) / dim_);
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) return;

    seqlen *= 2;
    seq_len_cached_ = seqlen;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(inv_freq_.device());
    inv_freq_ =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, options) / dim_);
    auto seq = torch::arange(seqlen, options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(0, 0, seqlen);
  }

 private:
  int dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_VisionRotaryEmbedding);

class Qwen3OmniMoe_Thinker_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_VisionPatchMergerImpl(
      const ModelContext& context,
      bool use_postshuffle_norm = false) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();
    int64_t d_model = model_args.mm_projection_dim();
    int context_dim = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();
    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
    use_postshuffle_norm_ = use_postshuffle_norm;
    if (use_postshuffle_norm) {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    } else {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    }

    norm_->weight.set_data(norm_->weight.to(options));
    norm_->bias.set_data(norm_->bias.to(options));

    auto fc1 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true));
    fc1->weight.set_data(fc1->weight.to(options));
    fc1->bias.set_data(fc1->bias.to(options));
    auto act = torch::nn::GELU();
    auto fc2 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, d_model).bias(true));
    fc2->weight.set_data(fc2->weight.to(options));
    fc2->bias.set_data(fc2->bias.to(options));
    mlp_ = register_module("mlp", torch::nn::Sequential(fc1, act, fc2));
    layers_ = std::make_tuple(fc1, act, fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    if (use_postshuffle_norm_) {
      x = norm_(x.view({-1, hidden_size_}));
    } else {
      x = norm_(x).view({-1, hidden_size_});
    }
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm
    const auto& norm_dict = state_dict.get_dict_with_prefix("ln_q.");
    const auto& norm_weight = norm_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(norm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      norm_->weight.data().copy_(norm_weight);
      is_norm_weight_loaded_ = true;
    }
    const auto norm_bias = norm_dict.get_tensor("bias");
    if (norm_bias.defined()) {
      CHECK_EQ(norm_->bias.sizes(), norm_bias.sizes())
          << "bias size mismatch for " << name();
      norm_->bias.data().copy_(norm_bias);
      is_norm_bias_loaded_ = true;
    }

    const auto& fc1_dict = state_dict.get_dict_with_prefix("mlp.0.");
    const auto& fc1_weight = fc1_dict.get_tensor("weight");
    if (fc1_weight.defined()) {
      CHECK_EQ(std::get<0>(layers_)->weight.sizes(), fc1_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<0>(layers_)->weight.data().copy_(fc1_weight);
      is_fc1_weight_loaded_ = true;
    }
    const auto fc1_bias = fc1_dict.get_tensor("bias");
    if (fc1_bias.defined()) {
      CHECK_EQ(std::get<0>(layers_)->bias.sizes(), fc1_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<0>(layers_)->bias.data().copy_(fc1_bias);
      is_fc1_bias_loaded_ = true;
    }

    const auto& fc2_dict = state_dict.get_dict_with_prefix("mlp.2.");
    const auto& fc2_weight = fc2_dict.get_tensor("weight");
    if (fc2_weight.defined()) {
      CHECK_EQ(std::get<2>(layers_)->weight.sizes(), fc2_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<2>(layers_)->weight.data().copy_(fc2_weight);
      is_fc2_weight_loaded_ = true;
    }
    const auto fc2_bias = fc2_dict.get_tensor("bias");
    if (fc2_bias.defined()) {
      CHECK_EQ(std::get<2>(layers_)->bias.sizes(), fc2_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<2>(layers_)->bias.data().copy_(fc2_bias);
      is_fc2_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_fc1_weight_loaded_)
        << "weight is not loaded for " << prefix + "mlp.0." + ".weight";
    CHECK(is_fc1_bias_loaded_)
        << "bias is not loaded for " << prefix + "mlp.0." + ".bias";
    CHECK(is_fc2_weight_loaded_)
        << "weight is not loaded for " << prefix + "mlp.2." + ".weight";
    CHECK(is_fc2_bias_loaded_)
        << "bias is not loaded for " << prefix + "mlp.2." + ".bias";
    CHECK(is_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "ln_q" + ".weight";
    CHECK(is_norm_bias_loaded_)
        << "bias is not loaded for " << prefix + "ln_q" + ".bias";
  }

 private:
  int hidden_size_;
  bool use_postshuffle_norm_;
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<torch::nn::Linear, torch::nn::GELU, torch::nn::Linear> layers_ = {
      nullptr,
      nullptr,
      nullptr};
  bool is_fc1_weight_loaded_ = false;
  bool is_fc1_bias_loaded_ = false;
  bool is_fc2_weight_loaded_ = false;
  bool is_fc2_bias_loaded_ = false;
  bool is_norm_weight_loaded_ = false;
  bool is_norm_bias_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_VisionPatchMerger);

class Qwen3OmniMoe_Thinker_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_VisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    auto& visual_indexes = model_args.mm_deepstack_visual_indexes();
    deepstack_visual_indexes_.insert(deepstack_visual_indexes_.end(),
                                     visual_indexes.begin(),
                                     visual_indexes.end());
    image_size_ = model_args.mm_image_size();
    spatial_merge_unit_ =
        static_cast<int>(spatial_merge_size_ * spatial_merge_size_);
    num_grid_per_side_ = image_size_ / patch_size_;

    patch_embed_ = register_module(
        "patch_embed", Qwen3OmniMoe_Thinker_VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module(
        "rotary_pos_emb", Qwen3OmniMoe_Thinker_VisionRotaryEmbedding(context));

    blocks_ = register_module("blocks", torch::nn::ModuleList());
    deepstack_mergers_ =
        register_module("deepstack_mergers", torch::nn::ModuleList());

    emb_ = register_module(
        "embedding",
        torch::nn::Embedding(static_cast<int>(std::pow(num_grid_per_side_, 2)),
                             hidden_size_));
    emb_->weight.set_data(emb_->weight.to(options_));

    merger_ = register_module("merger",
                              Qwen3OmniMoe_Thinker_VisionPatchMerger(context));

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = Qwen3OmniMoe_Thinker_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    for (int32_t idx = 0; idx < deepstack_visual_indexes_.size(); idx++) {
      auto merger = Qwen3OmniMoe_Thinker_VisionPatchMerger(context, true);
      deepstack_mergers_->push_back(merger);
      deepstack_merger_layers_.push_back(merger);
    }
  }

  torch::Tensor rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    auto count = grid_thw.sizes()[0];
    pos_ids_vec.reserve(count);

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto t = grid_thw_cpu[idx][0].item<int64_t>();
      auto h = grid_thw_cpu[idx][1].item<int64_t>();
      auto w = grid_thw_cpu[idx][2].item<int64_t>();

      auto hpos_ids = torch::arange(h, options).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      auto wpos_ids = torch::arange(w, options).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      pos_ids_vec.push_back(
          torch::stack({hpos_ids, wpos_ids}, -1).repeat({t, 1}));
    }

    auto pos_ids = torch::cat(pos_ids_vec, 0);
    auto max_grid_size =
        grid_thw
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None)})
            .max();

    auto rotary_pos_emb_full = rotary_pos_emb_(max_grid_size.item<int64_t>());
    auto rotary_pos_emb = rotary_pos_emb_full.index({pos_ids}).flatten(1);

    return rotary_pos_emb;
  }

  torch::Tensor fast_pos_embed_interpolate(const torch::Tensor& grid_thw) {
    auto device = grid_thw.device();
    int64_t hidden_dim = hidden_size_;
    int64_t m_size = spatial_merge_size_;

    auto grid_cpu = grid_thw.to(torch::kCPU);
    int64_t count = grid_thw.size(0);

    std::vector<torch::Tensor> outputs;
    outputs.reserve(count);

    for (int64_t idx = 0; idx < count; ++idx) {
      int64_t t = grid_cpu[idx][0].item<int64_t>();
      int64_t h = grid_cpu[idx][1].item<int64_t>();
      int64_t w = grid_cpu[idx][2].item<int64_t>();

      auto h_idxs =
          torch::linspace(
              0, static_cast<float>(num_grid_per_side_ - 1), h, torch::kFloat32)
              .to(device);
      auto w_idxs =
          torch::linspace(
              0, static_cast<float>(num_grid_per_side_ - 1), w, torch::kFloat32)
              .to(device);

      auto h_floor = h_idxs.to(torch::kLong);
      auto w_floor = w_idxs.to(torch::kLong);
      auto h_ceil = torch::clamp(h_floor + 1, 0, num_grid_per_side_ - 1);
      auto w_ceil = torch::clamp(w_floor + 1, 0, num_grid_per_side_ - 1);

      auto dh = h_idxs - h_floor;
      auto dw = w_idxs - w_floor;

      auto mesh_d = torch::meshgrid({dh, dw}, "ij");
      auto dh_grid = mesh_d[0], dw_grid = mesh_d[1];

      auto mesh_floor = torch::meshgrid({h_floor, w_floor}, "ij");
      auto h_floor_grid = mesh_floor[0];
      auto w_floor_grid = mesh_floor[1];

      auto mesh_ceil = torch::meshgrid({h_ceil, w_ceil}, "ij");
      auto h_ceil_grid = mesh_ceil[0];
      auto w_ceil_grid = mesh_ceil[1];

      auto h_floor_grid_idx = h_floor_grid * num_grid_per_side_;
      auto h_ceil_grid_idx = h_ceil_grid * num_grid_per_side_;

      auto w11 = dh_grid * dw_grid;
      auto w10 = dh_grid - w11;
      auto w01 = dw_grid - w11;
      auto w00 = 1.0f - dh_grid - dw_grid + w11;

      auto idx00 = h_floor_grid_idx + w_floor_grid;
      auto idx01 = h_floor_grid_idx + w_ceil_grid;
      auto idx10 = h_ceil_grid_idx + w_floor_grid;
      auto idx11 = h_ceil_grid_idx + w_ceil_grid;

      auto indices = torch::stack({idx00, idx01, idx10, idx11}, 0)
                         .reshape({4, -1})
                         .to(torch::kLong);
      auto weights = torch::stack({w00, w01, w10, w11}, 0)
                         .reshape({4, -1, 1})
                         .to(options_);

      auto embeds = emb_(indices);

      auto combined = (embeds * weights).sum(0);  // [h*w, hidden_dim]

      auto repeated = combined.unsqueeze(0).expand({t, -1, -1}).contiguous();
      repeated = repeated.view(
          {t, h / m_size, m_size, w / m_size, m_size, hidden_dim});
      repeated = repeated.permute({0, 1, 3, 2, 4, 5}).reshape({-1, hidden_dim});

      outputs.push_back(repeated);
    }

    return torch::cat(outputs, 0);
  }

  std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(
      torch::Tensor hidden_states,
      torch::Tensor grid_thw,  // [batch,thw]
      const ModelInputParams& input_params) {
    hidden_states = patch_embed_(hidden_states);
    auto pos_embeds = fast_pos_embed_interpolate(grid_thw);
    hidden_states = hidden_states + pos_embeds;
    //   compute position embedding
    auto rotary_pos_emb = rot_pos_emb(grid_thw);
    // compute cu_seqlens
    auto cu_seqlens = torch::repeat_interleave(
                          grid_thw.index({torch::indexing::Slice(), 1}) *
                              grid_thw.index({torch::indexing::Slice(), 2}),
                          grid_thw.index({torch::indexing::Slice(), 0}))
                          .cumsum(0, torch::kInt32);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));

    // transformers
    cu_seqlens = torch::diff(cu_seqlens);

    m_cos_ = rotary_pos_emb.cos().type_as(hidden_states);
    m_cos_ = m_cos_.repeat({1, 2});
    m_sin_ = rotary_pos_emb.sin().type_as(hidden_states);
    m_sin_ = m_sin_.repeat({1, 2});

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int>(),  // full seqlen vec
        cu_seqlens_cpu.data_ptr<int>() + cu_seqlens_cpu.numel());
    std::vector<torch::Tensor> deepstack_feature_lists;
    deepstack_feature_lists.reserve(deepstack_visual_indexes_.size());
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      hidden_states = layers_[idx](hidden_states,
                                   m_cos_,
                                   m_sin_,
                                   cu_seqlens,
                                   cu_seqlens_vec,
                                   input_params_new,
                                   idx);
      auto it = std::find(deepstack_visual_indexes_.begin(),
                          deepstack_visual_indexes_.end(),
                          idx);

      if (it != deepstack_visual_indexes_.end()) {
        int index = std::distance(deepstack_visual_indexes_.begin(), it);
        deepstack_feature_lists.push_back(
            deepstack_merger_layers_[index](hidden_states));
      }
    }
    // adapter
    hidden_states = merger_(hidden_states);
    return std::make_tuple(hidden_states, deepstack_feature_lists);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));

    for (int idx = 0; idx < deepstack_merger_layers_.size(); ++idx) {
      deepstack_merger_layers_[idx]->load_state_dict(
          state_dict.get_dict_with_prefix("merger_list." + std::to_string(idx) +
                                          "."));
    }

    const auto& emb_dict = state_dict.get_dict_with_prefix("pos_embed.");
    const auto& emb_weight = emb_dict.get_tensor("weight");
    if (emb_weight.defined()) {
      CHECK_EQ(emb_->weight.sizes(), emb_weight.sizes())
          << "weight size mismatch for " << name();
      emb_->weight.data().copy_(emb_weight);
      is_emb_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    merger_->verify_loaded_weights(prefix + "merger.");

    for (int idx = 0; idx < deepstack_merger_layers_.size(); ++idx) {
      deepstack_merger_layers_[idx]->verify_loaded_weights(
          "deepstack_merger_list." + std::to_string(idx) + ".");
    }
    CHECK(is_emb_weight_loaded_)
        << "weight is not loaded for " << prefix + "" + ".bias";
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
  }

 private:
  int hidden_size_ = 0;
  int num_heads_ = 0;
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::vector<int64_t> deepstack_visual_indexes_;
  int spatial_merge_unit_ = 0;
  int64_t image_size_ = 0;
  int num_grid_per_side_ = 0;

  Qwen3OmniMoe_Thinker_VisionPatchEmbed patch_embed_{nullptr};
  Qwen3OmniMoe_Thinker_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::Embedding emb_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3OmniMoe_Thinker_VisionBlock> layers_;

  torch::nn::ModuleList deepstack_mergers_{nullptr};
  std::vector<Qwen3OmniMoe_Thinker_VisionPatchMerger> deepstack_merger_layers_;
  Qwen3OmniMoe_Thinker_VisionPatchMerger merger_{nullptr};

  torch::Tensor m_cos_;
  torch::Tensor m_sin_;
  int device_id_ = 0;
  bool is_emb_weight_loaded_ = false;
  torch::TensorOptions options_;
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_VisionTransformer);

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

struct Qwen3_OmniAudioInputs {
  torch::Tensor input_features;
  torch::Tensor feat_length;
  torch::Tensor feat_origin_lens;
};

class Qwen3OmniMoe_Thinker_ForConditionalGenerationImpl
    : public torch::nn::Module {
 public:
  Qwen3OmniMoe_Thinker_ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual",
                              Qwen3OmniMoe_Thinker_VisionTransformer(context));
    audio_tower_ = register_module(
        "audio_tower", Qwen3OmniMoe_Thinker_AudioTransformer(context));
    language_model_ =
        register_module("language_model", Qwen3MoeForCausalLM(context));
  }

  void prepare_encoder_input(
      const ModelInputParams& input_params,
      std::optional<Qwen3_VLImageInputs>& image_inputs,
      std::optional<Qwen3_VLVideoInputs>& video_inputs,
      std::optional<Qwen3_OmniAudioInputs>& audio_inputs) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = res.value();
    }

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    torch::Tensor second_per_grid_ts;
    if (const auto& res = mm_data.get<torch::Tensor>("second_per_grid_ts")) {
      second_per_grid_ts = res.value();
    }

    torch::Tensor input_features;
    if (const auto& res = mm_data.get<torch::Tensor>("input_features")) {
      input_features = res.value();
    }

    torch::Tensor feat_length;
    if (const auto& res = mm_data.get<torch::Tensor>("feat_length")) {
      feat_length = res.value();
    }

    torch::Tensor feat_origin_lens;
    if (const auto& res = mm_data.get<torch::Tensor>("feat_origin_lens")) {
      feat_origin_lens = res.value();
    }

    if (pixel_values.defined() && image_grid_thw.defined()) {
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};
    }

    if (pixel_values_videos.defined() && video_grid_thw.defined() &&
        second_per_grid_ts.defined()) {
      video_inputs = Qwen3_VLVideoInputs{
          pixel_values_videos, video_grid_thw, second_per_grid_ts};
    }

    if (input_features.defined() && feat_length.defined() &&
        feat_origin_lens.defined()) {
      audio_inputs =
          Qwen3_OmniAudioInputs{input_features, feat_length, feat_origin_lens};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    std::optional<Qwen3_OmniAudioInputs> audio_input;
    prepare_encoder_input(input_params, image_input, video_input, audio_input);
    MMDict multimodal_embeds;
    auto merge_size = model_args_.mm_image_merge_size();
    string prefix = FLAGS_use_audio_in_video ? "" : "embedding|";
    if (image_input) {
      auto [image_embeds, deep_stacks] =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()),
                  input_params);

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);
      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, 0 /*dim*/);
      string image_prefix = "image|" + prefix + "deepstack_";
      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[image_prefix + std::to_string(i)] =
            deep_stacks[i].split(image_tokens_vec, 0 /*dim*/);
      }
    }
    if (video_input) {
      auto [video_embeds, deep_stacks] =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()),
                  input_params);
      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);
      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      multimodal_embeds["video|embedding"] =
          video_embeds.split(video_tokens_vec, 0 /*dim*/);
      string video_prefix = "video|" + prefix + "deepstack_";
      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[video_prefix + std::to_string(i)] =
            deep_stacks[i].split(video_tokens_vec, 0 /*dim*/);
      }
    }
    if (audio_input) {
      auto feat_origin_lens =
          audio_input->feat_origin_lens.to(options_.device(), torch::kLong);

      auto input_features =
          audio_input->input_features.permute({1, 0}).to(options_);

      auto audio_embeds =
          audio_tower_->forward(input_features, input_params, feat_origin_lens);

      auto audio_tokens =
          audio_input->feat_length.cpu().contiguous().to(torch::kLong);

      std::vector<int64_t> feature_lens_vec(
          audio_tokens.data_ptr<int64_t>(),
          audio_tokens.data_ptr<int64_t>() + audio_tokens.numel());

      multimodal_embeds["audio|embedding"] =
          audio_embeds.split(feature_lens_vec, 0 /*dim*/);
    }
    if (FLAGS_use_audio_in_video && video_input && audio_input) {
      auto origin_audio_embeds = std::get<std::vector<torch::Tensor>>(
          multimodal_embeds["audio|embedding"]);
      auto origin_video_embeds = std::get<std::vector<torch::Tensor>>(
          multimodal_embeds["video|embedding"]);
      std::vector<torch::Tensor> audio_in_video_embeds;
      audio_in_video_embeds.reserve(origin_video_embeds.size());
      std::vector<torch::Tensor> audio_embeds;
      audio_embeds.reserve(origin_audio_embeds.size() -
                           origin_video_embeds.size());
      size_t audio_index = 0;
      size_t video_index = 0;

      const auto& mm_data = input_params.mm_data;
      const auto& mm_data_vec = mm_data.mm_data_vec();
      for (const auto mm_data : mm_data_vec) {
        const auto& mm_items = mm_data.items<MMItemVec>();
        for (const auto& item : mm_items) {
          if (item.type() & MMType::AUDIO) {
            audio_embeds.emplace_back(origin_audio_embeds[audio_index++]);
          } else if (item.type() & MMType::VIDEO) {
            auto video_embed = origin_video_embeds[video_index++];
            auto audio_embed = origin_audio_embeds[audio_index++];
            torch::Tensor audio_in_video_mask;
            auto res = item.get<torch::Tensor>("audio_in_video_mask");
            if (res.has_value()) {
              audio_in_video_mask = res.value();
            }
            torch::Tensor audio_in_video_embedding = torch::full(
                {audio_in_video_mask.size(0), origin_video_embeds[0].size(1)},
                1,
                options_);
            auto video_mask =
                torch::isin(audio_in_video_mask, model_args_.video_token_id());
            auto audio_mask =
                torch::isin(audio_in_video_mask, model_args_.audio_token_id());
            audio_in_video_embedding.index_put_({video_mask}, video_embed);
            audio_in_video_embedding.index_put_({audio_mask}, audio_embed);
            audio_in_video_embeds.emplace_back(audio_in_video_embedding);
          }
        }
      }
      multimodal_embeds["audio|embedding"] = audio_embeds;
      multimodal_embeds["video|embedding"] = audio_in_video_embeds;
    }
    return multimodal_embeds;
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    auto special_token_ids = torch::tensor(
        {model_args_.image_token_id(), model_args_.video_token_id()},
        input_ids.options().dtype(torch::kInt64));
    auto is_multimodal = torch::isin(input_ids, special_token_ids);
    return is_multimodal;
  }

  torch::Tensor generate_multimodal_mask_with_audio(torch::Tensor input_ids) {
    auto special_token_ids =
        torch::tensor({model_args_.image_token_id(),
                       model_args_.video_token_id(),
                       model_args_.audio_token_id()},
                      input_ids.options().dtype(torch::kInt64));
    auto is_multimodal = torch::isin(input_ids, special_token_ids);
    return is_multimodal;
  }

  std::vector<torch::Tensor> get_deep_stacks(
      const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    string prefix = FLAGS_use_audio_in_video ? "" : "embedding|";
    if (!mm_data.has(prefix + "deepstack_0")) {
      return {};
    }

    std::vector<torch::Tensor> deepstacks = {
        mm_data.get<torch::Tensor>(prefix + "deepstack_0").value(),
        mm_data.get<torch::Tensor>(prefix + "deepstack_1").value(),
        mm_data.get<torch::Tensor>(prefix + "deepstack_2").value()};
    return deepstacks;
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor multimodal_embeds;
    if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }
    auto is_multimodal = generate_multimodal_mask(input_ids);
    auto is_multimodal_with_audio =
        generate_multimodal_mask_with_audio(input_ids);
    input_params.visual_pos_masks = is_multimodal;
    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal_with_audio);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    input_params.deep_stacks = std::move(get_deep_stacks(input_params));
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      if (state_dict->get_tensor("thinker.lm_head.weight").defined()) {
        state_dict->rename_prefix_inplace("thinker.lm_head.", "lm_head.");
      }
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.visual."));
      audio_tower_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.audio_tower."));
    }
    // verify
    visual_->verify_loaded_weights("thinker.visual.");
    visual_->merge_loaded_weights();
    audio_tower_->verify_loaded_weights("thinker.audio_tower.");
    audio_tower_->merge_loaded_weights();
    audio_tower_->to(options_.device(),
                     torch::typeMetaToScalarType(options_.dtype()));

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader), "thinker.model.");
    }
  }

  layer::NpuLmHead get_npu_lm_head() {
    return language_model_->get_npu_lm_head();
  }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    language_model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3OmniMoe_Thinker_VisionTransformer visual_{nullptr};
  Qwen3OmniMoe_Thinker_AudioTransformer audio_tower_{nullptr};
  Qwen3MoeForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_Thinker_ForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen3_omni_moe_thinker, Qwen3OmniMoe_InputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe_thinker,
                          Qwen3OmniMoe_Thinker_ForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_omni_moe_thinker, Qwen2VLImageProcessor);
REGISTER_FEATURE_EXTRACTOR(qwen3_omni_moe_thinker, WhisperFeatureExtractor);

REGISTER_MODEL_ARGS(qwen3_omni_moe_thinker, [&] {
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
