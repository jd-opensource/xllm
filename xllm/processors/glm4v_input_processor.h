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

#pragma once

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "multimodal_input_processor.h"

namespace xllm {

class Glm4VInputProcessor : public MultimodalInputProcessor {
 public:
  Glm4VInputProcessor(const ModelArgs&);
  ~Glm4VInputProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);
  bool process_videos(std::vector<torch::Tensor> videos,
                      std::vector<VideoMetadata> video_meta_list,
                      MMData& mm_datas);
  bool process_video(torch::Tensor video,
                     VideoMetadata& metadata,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);
  torch::Tensor sample_frames(const VideoMetadata& metadata,
                              int32_t temporal_patch_size);

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;

  bool do_rescale_ = true;
  bool do_resize_ = true;

  std::vector<double> image_mean_;
  std::vector<double> image_std_;

  int32_t max_pixels_ = 12845056;
  int32_t min_pixels_ = 3136;

  int32_t merge_size_ = 2;
  int32_t patch_size_ = 14;

  std::vector<double> video_mean_;
  std::vector<double> video_std_;

  int32_t video_max_pixels_ = 47040000;
  int32_t video_min_pixels_ = 12544;

  int32_t video_merge_size_ = 2;
  int32_t video_patch_size_ = 14;

  int32_t resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  std::unordered_map<std::string, int32_t> size_;
  int32_t temporal_patch_size_ = 2;
  int32_t video_temporal_patch_size_ = 2;

  bool do_sample_frame_ = true;

  int32_t min_frames_ = 4;
  int32_t max_frames_ = 768;
};

}  // namespace xllm
