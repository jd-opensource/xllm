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
#include <torch/torch.h>

#include <cmath>
#include <string>
#include <vector>

namespace xllm {

inline torch::Tensor get_feat_extract_output_lengths(
    torch::Tensor input_lengths) {
  auto input_lengths_leave = input_lengths % 100;
  auto feat_lengths = torch::floor_divide((input_lengths_leave - 1), 2) + 1;
  auto output_lengths =
      torch::floor_divide((torch::floor_divide((feat_lengths - 1), 2) + 1 - 1),
                          2) +
      1 + torch::floor_divide(input_lengths, 100) * 13;
  return output_lengths.to(torch::kInt64);
}

inline torch::Tensor hertz_to_mel(const torch::Tensor& freq,
                                  const std::string& mel_scale = "htk") {
  if (mel_scale != "slaney" && mel_scale != "htk" && mel_scale != "kaldi") {
    CHECK(false) << "mel_scale should be one of 'htk', 'slaney' or 'kaldi'.";
  }

  if (mel_scale == "htk") {
    return 2595.0 * torch::log10(1.0f + freq / 700.0);
  } else if (mel_scale == "kaldi") {
    const float kaldi_scale = 1127.0;
    return kaldi_scale * torch::log1p(freq / 700.0);
  }

  double min_log_hertz = 1000.0;
  double min_log_mel = 15.0;
  double logstep = 27.0 / std::log(6.4);

  auto mels = 3.0 * freq / 200.0;

  auto result =
      torch::where(freq >= min_log_hertz,
                   min_log_mel + torch::log(freq / min_log_hertz) * logstep,
                   mels);

  return result;
}

inline torch::Tensor mel_to_hertz(const torch::Tensor& mels,
                                  const std::string& mel_scale = "htk") {
  if (mel_scale != "slaney" && mel_scale != "htk" && mel_scale != "kaldi") {
    CHECK(false) << "mel_scale should be one of 'htk', 'slaney' or 'kaldi'.";
  }

  if (mel_scale == "htk") {
    return 700.0 * (torch::pow(10.0, mels / 2595.0) - 1.0);
  } else if (mel_scale == "kaldi") {
    const float kaldi_scale = 1127.0;
    return 700.0 * (torch::exp(mels / kaldi_scale) - 1.0);
  }

  const double min_log_hertz = 1000.0;
  const double min_log_mel = 15.0;
  const double logstep = std::log(6.4) / 27.0;

  auto freq = 200 * mels / 3.0;

  auto result =
      torch::where(mels >= min_log_mel,
                   min_log_hertz * torch::exp(logstep * (mels - min_log_mel)),
                   freq);

  return result;
}

inline torch::Tensor create_triangular_filter_bank(
    const torch::Tensor& fft_freqs,
    const torch::Tensor& filter_freqs) {
  // fft_freqs: [num_frequency_bins]
  // filter_freqs: [num_mel_filters]

  auto filter_diff = torch::diff(filter_freqs);

  auto fft_freqs_expanded = fft_freqs.unsqueeze(1);
  auto filter_freqs_expanded = filter_freqs.unsqueeze(0);

  auto slopes = filter_freqs_expanded - fft_freqs_expanded;
  auto down_slopes = -slopes.slice(1, 0, -2) / filter_diff.slice(0, 0, -1);
  auto up_slopes = slopes.slice(1, 2) / filter_diff.slice(0, 1);

  auto mel_filters = torch::minimum(down_slopes, up_slopes);
  mel_filters = torch::clamp_min(mel_filters, 0.0f);

  return mel_filters;
}

inline torch::Tensor mel_filter_bank(int64_t num_frequency_bins,
                                     int64_t num_mel_filters,
                                     double min_frequency,
                                     double max_frequency,
                                     int64_t sampling_rate,
                                     const std::string& norm = "",
                                     const std::string& mel_scale = "htk",
                                     bool triangularize_in_mel_space = false) {
  if (!norm.empty() && norm != "slaney") {
    CHECK(false) << "norm must be one of empty string or 'slaney'";
  }

  if (num_frequency_bins < 2) {
    CHECK(false) << "Require num_frequency_bins: " +
                        std::to_string(num_frequency_bins) + " >= 2";
  }

  if (min_frequency > max_frequency) {
    CHECK(false) << "Require min_frequency: " + std::to_string(min_frequency) +
                        " <= max_frequency: " + std::to_string(max_frequency);
  }

  if (max_frequency > sampling_rate / 2.0) {
    CHECK(false)
        << "Warning: max_frequency exceeds Nyquist frequency (sampling_rate/2)";
  }

  auto mel_min_scalar = hertz_to_mel(torch::tensor(min_frequency), mel_scale);
  auto mel_max_scalar = hertz_to_mel(torch::tensor(max_frequency), mel_scale);
  double mel_min = mel_min_scalar.item<double>();
  double mel_max = mel_max_scalar.item<double>();

  auto mel_freqs = torch::linspace(mel_min, mel_max, num_mel_filters + 2);
  auto filter_freqs = mel_to_hertz(mel_freqs, mel_scale);
  torch::Tensor fft_freqs;

  if (triangularize_in_mel_space) {
    float fft_bin_width =
        static_cast<float>(sampling_rate) / ((num_frequency_bins - 1) * 2);
    auto indices = torch::arange(num_frequency_bins, torch::kFloat32);
    fft_freqs = hertz_to_mel(fft_bin_width * indices, mel_scale);
    filter_freqs = mel_freqs;
  } else {
    fft_freqs = torch::linspace(0, sampling_rate / 2, num_frequency_bins);
  }

  auto mel_filters = create_triangular_filter_bank(fft_freqs, filter_freqs);

  if (!norm.empty() && norm == "slaney") {
    auto filter_widths = filter_freqs.slice(0, 2, num_mel_filters + 2) -
                         filter_freqs.slice(0, 0, num_mel_filters);
    auto enorm = 2.0 / filter_widths;
    mel_filters *= enorm.unsqueeze(0);
  }

  return mel_filters;
}

}  // namespace xllm
