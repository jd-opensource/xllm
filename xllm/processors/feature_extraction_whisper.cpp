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

#include "feature_extraction_whisper.h"

#include "core/framework/state_dict/utils.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {

WhisperFeatureExtractor::WhisperFeatureExtractor(const ModelArgs& args)
    : FeatureExtractor(args.mm_audio_feature_size(),
                       args.mm_audio_sampling_rate(),
                       args.mm_audio_padding_value(),
                       args.mm_audio_padding_side(),
                       args.mm_audio_return_attention_mask()),
      n_fft_(args.mm_audio_n_fft()),
      hop_length_(args.mm_audio_hop_length()),
      chunk_length_(args.mm_audio_chunk_length()),
      n_samples_(args.mm_audio_chunk_length() * args.mm_audio_sampling_rate()),
      sampling_rate_(args.mm_audio_sampling_rate()),
      dither_(args.mm_audio_dither()) {
  nb_max_frames_ = n_samples_ / args.mm_audio_hop_length();
  model_input_names_ = {"input_features"};

  mel_filters_ = audio_utils::mel_filter_bank(1 + n_fft_ / 2,
                                              feature_size_,
                                              0.0f,
                                              8000.0f,
                                              sampling_rate_,
                                              "slaney",
                                              "slaney");

  options_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  truncation_ = args.mm_audio_truncation();
  do_normalize_ = args.mm_audio_do_normalize();
  return_token_timestamps_ = args.mm_audio_return_token_timestamps();
  return_attention_mask_ = args.mm_audio_return_attention_mask();
  use_audio_in_video_ = args.mm_use_audio_in_video();
  padding_ = static_cast<PaddingStrategy>(args.mm_audio_padding_strategy());

  if (args.mm_audio_pad_to_multiple_of() > 0) {
    pad_to_multiple_of_ = args.mm_audio_pad_to_multiple_of();
  }

  if (args.mm_audio_max_length() > 0) {
    max_length_ = args.mm_audio_max_length();
  }
}

/*
Args:
   waveform : torch::Tensor
        is expected to be a batch of tensor (feature_len, n),
*/
torch::Tensor WhisperFeatureExtractor::torch_extract_fbank_features_(
    const torch::Tensor& waveform,
    const torch::TensorOptions& options) {
  CHECK(options.has_device() && options.has_dtype())
      << "tensor options of WhisperFeatureExtractor has not been initialized";

  auto waveform_tensor = waveform.to(options.device(), torch::kFloat32);

  auto window = torch::hann_window(n_fft_, true, options);

  if (dither_ != 0.0) {
    auto noise =
        torch::randn(waveform_tensor.sizes(), waveform_tensor.options());
    waveform_tensor = waveform_tensor + dither_ * noise;
  }

  auto stft = torch::stft(waveform_tensor,
                          n_fft_,
                          hop_length_,
                          n_fft_,
                          window,
                          true,
                          "reflect",
                          false,
                          c10::nullopt,
                          true);

  auto magnitudes = stft.slice(-1, 0, -1).abs().pow(2);

  auto mel_filters_device = mel_filters_.to(waveform_tensor.options());
  auto mel_spec = torch::matmul(mel_filters_device.t(), magnitudes);

  auto log_spec = torch::clamp(mel_spec, 1e-10).log10();

  if (waveform_tensor.dim() == 2) {
    auto max_val = log_spec.amax({2}, true).amax({1}, true);
    log_spec = torch::maximum(log_spec, max_val - 8.0);
  } else {
    auto max_val = log_spec.max();
    log_spec = torch::maximum(log_spec, max_val - 8.0);
  }

  log_spec = (log_spec + 4.0) / 4.0;

  return log_spec;
}

/*
Args:
    batched_feature: BatchFeature
         A data class that contains input_features and masks
    input_features: vector<torch::Tensor>
         input_feature is expected to be a vector<torch::Tensor> (tensor, 1),
         with tensor shape [n, feature_len, 1]
    masks: optional<vector<torch::Tensor>>
         should have the same shape as input_features
*/
void WhisperFeatureExtractor::zero_mean_unit_var_norm_(
    BatchFeature& batched_feature,
    double padding_value) {
  const torch::Tensor& input_values =
      batched_feature.get<std::vector<torch::Tensor>>("input_features")
          .value()[0];

  const auto& attention_mask_vector =
      batched_feature.get<std::vector<torch::Tensor>>("attention_mask");
  torch::Tensor attention_mask;
  if (attention_mask_vector.has_value() &&
      !attention_mask_vector.value().empty()) {
    attention_mask = attention_mask_vector.value()[0];
  }

  std::vector<torch::Tensor> normalized;

  for (size_t i = 0; i < input_values.size(0); i++) {
    auto vector = input_values[i];

    if (attention_mask.defined()) {
      auto mask = attention_mask[i];
      auto length = mask.sum().item<int64_t>();

      CHECK(length > 0)
          << "mask lenght is expected to be greater than 0, but got: "
          << length;
      auto slice = vector.narrow(0, 0, length);
      auto mean_val = slice.mean();
      auto var_val = slice.var(false);

      auto normed_slice = (vector - mean_val) / torch::sqrt(var_val + 1e-7f);

      if (length < vector.size(0)) {
        normed_slice.slice(0, length) = padding_value;
      }

      normalized.push_back(normed_slice);
    } else {
      auto mean_val = vector.mean();
      auto var_val = vector.var(false);
      auto normed = (vector - mean_val) / torch::sqrt(var_val + 1e-7f);
      normalized.push_back(normed);
    }
  }
  torch::Tensor stacked = torch::stack(normalized, 0);
  batched_feature.update("input_features", std::vector<torch::Tensor>{stacked});
}

/*
Args:
    raw_speech: vector<torch::Tensor>
        raw_speech is expected to be a batch of tensor,  vector<torch::Tensor>
        (tensor, n), with tensor shape [feature_len]
    BatchFeature.main_feature: torch::Tensor
        expected to be a  torch::Tensor, tensor with shape
        [feat_len, hidden_size],
    BatchFeature.feat_len: torch::Tensor
        expected to be a  torch::Tensor, tensor with shape [1],
    BatchFeature.num_frames: std::optional<torch::Tensor>
        expected to be a  torch::Tensor, tensor with shape [1]
*/
bool WhisperFeatureExtractor::process(const MMInput& mm_inputs,
                                      MMData& mm_datas) {
  std::vector<torch::Tensor> raw_speech;
  std::vector<torch::Tensor> raw_speech_in_video;
  for (const auto& input_item : mm_inputs) {
    // in use_audio_in_video case, input_item.type_ & MMType::VIDEO
    // and input_item.type_ & MMType::AUDIO could both be true
    // so we start with video type
    if (input_item.has_type(MMType::VIDEO) &&
        input_item.decode_audio.defined() && use_audio_in_video_) {
      raw_speech_in_video.push_back(input_item.decode_audio);
    } else if (input_item.has_type(MMType::AUDIO) &&
               input_item.decode_audio.defined()) {
      raw_speech.push_back(input_item.decode_audio);
    }
  }
  bool process_result = true;
  if (!raw_speech.empty()) {
    process_result = process_audio(mm_inputs, mm_datas, raw_speech, false);
  }
  if (!raw_speech_in_video.empty() && use_audio_in_video_) {
    process_result =
        process_result &
        process_audio(mm_inputs, mm_datas, raw_speech_in_video, true);
  }
  return process_result;
}

bool WhisperFeatureExtractor::process_audio(
    const MMInput& mm_inputs,
    MMData& mm_datas,
    std::vector<torch::Tensor> raw_speech,
    bool audio_in_video) {
  int64_t batch_size = raw_speech.size();

  if (raw_speech.empty()) {
    return true;
  }
  std::vector<torch::Tensor> batched_speech;
  for (const auto& speech : raw_speech) {
    if (speech.dim() > 1) {
      LOG(ERROR) << "Only mono-channel audio is supported";
    }

    torch::Tensor processed_speech;
    if (speech.dim() == 1) {
      processed_speech =
          speech.unsqueeze(0).t();  // tensor shape [feature_len, 1]
    } else {
      processed_speech = speech.t();
    }
    batched_speech.push_back(processed_speech);
  }
  BatchFeature batched_feature;
  batched_feature.add("input_features", batched_speech);

  pad(batched_feature,
      padding_,
      max_length_.has_value() ? max_length_.value() : n_samples_,
      truncation_,
      pad_to_multiple_of_,
      do_normalize_ || return_attention_mask_);

  if (do_normalize_) {
    zero_mean_unit_var_norm_(batched_feature, padding_value_.value());
  }

  torch::Tensor input_features;
  if (auto res =
          batched_feature.get<std::vector<torch::Tensor>>("input_features")) {
    input_features = res.value()[0];
  }

  input_features = input_features.permute({2, 0, 1});

  auto input_features_extracted =
      torch_extract_fbank_features_(input_features[0], options_);
  std::vector<torch::Tensor> input_features_extracted_vec{
      input_features_extracted};
  batched_feature.update("input_features", input_features_extracted_vec);

  torch::Tensor rescaled_attention_mask;
  if (return_attention_mask_) {
    torch::Tensor attention_mask;
    if (auto res =
            batched_feature.get<std::vector<torch::Tensor>>("attention_mask")) {
      attention_mask = res.value()[0];
    }

    rescaled_attention_mask = attention_mask.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, torch::indexing::None, hop_length_)});

    if (attention_mask.size(1) % hop_length_ != 0) {
      rescaled_attention_mask = rescaled_attention_mask.index(
          {torch::indexing::Slice(), torch::indexing::Slice(0, -1)});
    }

    batched_feature.update("attention_mask", rescaled_attention_mask);
  }

  std::vector<torch::Tensor> num_frames;
  if (return_token_timestamps_) {
    for (auto& speech : raw_speech) {
      num_frames.push_back(torch::tensor(
          static_cast<int64_t>(speech.size(0)) / hop_length_, torch::kInt32));
    }
    batched_feature.add("num_frames", num_frames);
  }

  MMItemPtrVec* audio_feature_items;

  if (audio_in_video) {
    // audio_in_video case, add extracted features to video items
    audio_feature_items = mm_datas.get(MMType::VIDEO);
  } else {
    audio_feature_items = mm_datas.get(MMType::AUDIO);
  }

  for (size_t feat_idx = 0; feat_idx < audio_feature_items->size();
       feat_idx++) {
    auto& item = audio_feature_items->at(feat_idx);
    if (!return_attention_mask_) {
      auto input_features = input_features_extracted[feat_idx].permute({1, 0});
      auto feat_origin_lens =
          torch::tensor({input_features.size(0)}, torch::kLong);
      auto feat_length =
          audio_utils::get_feat_extract_output_lengths(feat_origin_lens);
      item->set_data({{"input_features", input_features},
                      {"feat_length", feat_length},
                      {"feat_origin_lens", feat_origin_lens}});
    } else {
      torch::Tensor feat_origin_lens =
          torch::sum(rescaled_attention_mask[feat_idx], -1).to(torch::kLong);
      torch::Tensor feat_length =
          audio_utils::get_feat_extract_output_lengths(feat_origin_lens);
      // input_features has shape [hidden_size, seq_len], to make sure
      // that they could be batched together, permute it here and resever
      // the operation when inference
      auto permuted = input_features_extracted[feat_idx].permute({1, 0});
      auto bool_mask = rescaled_attention_mask[feat_idx].to(torch::kBool);
      auto input_features = permuted.index({bool_mask});
      if (audio_in_video) {
        item->add("input_features", input_features);
        item->add("feat_length",
                  torch::tensor({feat_length.item<int64_t>()}, torch::kLong));
        item->add(
            "feat_origin_lens",
            torch::tensor({feat_origin_lens.item<int64_t>()}, torch::kLong));
      } else {
        item->set_data(
            {{"input_features", input_features},
             {"feat_length",
              torch::tensor({feat_length.item<int64_t>()}, torch::kLong)},
             {"feat_origin_lens",
              torch::tensor({feat_origin_lens.item<int64_t>()},
                            torch::kLong)}});
      }
    }
  }

  return true;
}
}  // namespace xllm
