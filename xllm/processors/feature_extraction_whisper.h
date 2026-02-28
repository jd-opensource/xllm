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
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "feature_extractor.h"

namespace xllm {

class WhisperFeatureExtractor : public FeatureExtractor {
 public:
  WhisperFeatureExtractor(const ModelArgs& args);

  bool process(const MMInput& mm_inputs, MMData& mm_datas);

  bool process_audio(const MMInput& mm_inputs,
                     MMData& mm_datas,
                     std::vector<torch::Tensor> raw_speech,
                     bool audio_in_video);

  void set_options(torch::TensorOptions& options) { options_ = options; }

 private:
  torch::Tensor torch_extract_fbank_features_(
      const torch::Tensor& waveform,
      const torch::TensorOptions& options);

  void zero_mean_unit_var_norm_(BatchFeature& batch_feature,
                                double padding_value = 0.0);

  int64_t n_fft_;
  int64_t hop_length_;
  int64_t chunk_length_;
  int64_t n_samples_;      // chunk_length_ * sampling_rate_
  int64_t nb_max_frames_;  // n_samples_ // hop_length_
  int64_t sampling_rate_;
  double dither_;
  torch::TensorOptions options_;
  torch::Tensor mel_filters_;

  bool truncation_ = false;
  std::optional<int64_t> pad_to_multiple_of_ = std::nullopt;
  bool return_attention_mask_ = false;
  PaddingStrategy padding_ = PaddingStrategy::MAX_LENGTH;
  std::optional<int64_t> max_length_ = std::nullopt;
  bool do_normalize_ = false;
  bool return_token_timestamps_ = false;
  bool use_audio_in_video_ = false;
};

}  // namespace xllm
