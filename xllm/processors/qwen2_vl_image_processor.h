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

#include <tuple>
#include <unordered_map>
#include <vector>

#include "image_processor.h"

namespace xllm {

class Qwen2VLImageProcessor : public ImageProcessor {
 public:
  Qwen2VLImageProcessor(const ModelArgs&);
  ~Qwen2VLImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

  using Size = std::pair<int32_t, int32_t>;
  virtual std::optional<Size> smart_resize_image(int32_t height,
                                                 int32_t width,
                                                 int32_t factor,
                                                 int32_t min_pixels,
                                                 int32_t max_pixels) const;

  virtual std::optional<Size> smart_resize_video(int32_t num_frames,
                                                 int32_t height,
                                                 int32_t width,
                                                 int32_t temporal_factor,
                                                 int32_t factor,
                                                 int32_t min_pixels,
                                                 int32_t max_pixels) const;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);

  bool process_images_embedding(
      const std::vector<EmbeddingOutput>& images_embedding,
      MMData& mm_datas);

  bool process_videos(std::vector<torch::Tensor> videos,
                      std::vector<VideoMetadata> video_meta_list,
                      MMData& mm_datas);
  bool process_video(torch::Tensor video,
                     VideoMetadata& metadata,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);
  virtual torch::Tensor sample_frames(const VideoMetadata& metadata,
                                      int32_t temporal_patch_size,
                                      int32_t min_frames,
                                      int32_t max_frames,
                                      int32_t num_frames = -1,
                                      double set_fps = -1.0);

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

  int32_t resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  std::unordered_map<std::string, int> size_;
  int32_t temporal_patch_size_ = 2;

  bool do_sample_frame_ = true;

  int32_t min_frames_ = 4;
  int32_t max_frames_ = 768;
};

}  // namespace xllm
