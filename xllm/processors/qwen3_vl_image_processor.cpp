/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "qwen3_vl_image_processor.h"

namespace xllm {

std::optional<Qwen2VLImageProcessor::Size>
Qwen3VLImageProcessor::smart_resize_video(int num_frames,
                                          int height,
                                          int width,
                                          int temporal_factor,
                                          int factor,
                                          int min_pixels,
                                          int max_pixels) const {
  if (height < factor || width < factor) {
    LOG(ERROR) << "height:" << height << " or width:" << width
               << " must be larger than factor:" << factor;
    return std::nullopt;
  }
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200.0) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200";
    return std::nullopt;
  }

  int h_bar =
      static_cast<int>(std::round(height / static_cast<double>(factor))) *
      factor;
  int w_bar =
      static_cast<int>(std::round(width / static_cast<double>(factor))) *
      factor;
  int t_bar = static_cast<int>(std::ceil(
                  num_frames / static_cast<double>(temporal_factor))) *
              temporal_factor;

  const double thw_bar = static_cast<double>(t_bar) *
                         static_cast<double>(h_bar) *
                         static_cast<double>(w_bar);

  if (thw_bar > static_cast<double>(max_pixels)) {
    const double beta =
        std::sqrt((static_cast<double>(num_frames) * height * width) /
                  static_cast<double>(max_pixels));
    int h_new = static_cast<int>(
                    std::floor(height / beta / static_cast<double>(factor))) *
                factor;
    int w_new = static_cast<int>(
                    std::floor(width / beta / static_cast<double>(factor))) *
                factor;
    h_bar = std::max(factor, h_new);
    w_bar = std::max(factor, w_new);
  } else if (thw_bar < static_cast<double>(min_pixels)) {
    const double beta =
        std::sqrt(static_cast<double>(min_pixels) /
                  (static_cast<double>(num_frames) * height * width));
    h_bar = static_cast<int>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}

torch::Tensor Qwen3VLImageProcessor::sample_frames(
    const VideoMetadata& metadata,
    int /*temporal_patch_size*/,
    int min_frames,
    int max_frames,
    int num_frames,
    double set_fps) {
  if (set_fps > 0.0 && num_frames > 0) {
    LOG(FATAL) << "num_frames and fps are mutually exclusive arguments, please "
                  "use only one!";
  }

  double fps = set_fps;
  int total_num_frames = metadata.total_num_frames;

  if (num_frames <= 0 && fps > 0.0) {
    if (metadata.fps <= 0.0) {
      LOG(FATAL)
          << "Asked to sample `fps` frames per second but no video metadata "
             "was provided which is required when sampling with `fps`. ";
    }
    num_frames = static_cast<int>(static_cast<double>(total_num_frames) /
                                  metadata.fps * fps);
    num_frames = std::min(std::max(num_frames, min_frames),
                          std::min(max_frames, total_num_frames));
  }

  if (num_frames <= 0) {
    num_frames = std::min(std::max(total_num_frames, min_frames), max_frames);
  }

  auto lin = torch::linspace(0.0,
                             static_cast<double>(total_num_frames - 1),
                             num_frames,
                             torch::TensorOptions().dtype(torch::kFloat32));
  auto idx = torch::round(lin).to(torch::kLong);
  idx = torch::clamp(idx, 0, total_num_frames - 1);
  return idx;
}

}  // namespace xllm
