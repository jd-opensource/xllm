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

#include "clip_image_processor.h"

#include <memory>

namespace xllm {

CLIPImageProcessor::CLIPImageProcessor(const std::shared_ptr<ModelArgs>& args) {
  do_resize_ = args->mm_image_do_resize();
  do_center_crop_ = args->mm_image_do_center_crop();
  do_rescale_ = args->mm_image_do_rescale();
  do_normalize_ = args->mm_image_do_normalize();
  shortest_edge_ = args->mm_image_resize_shortest_edge();
  crop_size_ = std::make_pair(args->mm_image_crop_height_size(),
                              args->mm_image_crop_width_size());
  resample_ = args->mm_image_resample();
  rescale_factor_ = args->mm_image_rescale_factor();
  image_mean_ = args->mm_image_normalize_mean();
  image_std_ = args->mm_image_normalize_std();
}

bool CLIPImageProcessor::process(const MMInput& mm_inputs, MMData& mm_datas) {
  return false;
}

torch::Tensor CLIPImageProcessor::process_images(const torch::Tensor& images) {
  int batch_size = images.size(0);
  std::vector<torch::Tensor> processed_images;
  auto size = get_resize_output_image_size(images[0], shortest_edge_);

  for (int i = 0; i < batch_size; ++i) {
    torch::Tensor image = images[i];

    if (do_resize_) {
      image = resize(image, size, resample_);
    }

    if (do_center_crop_) {
      image = centerCrop(image, crop_size_);
    }

    if (do_rescale_) {
      image = rescale(image, rescale_factor_);
    }

    if (do_normalize_) {
      image = normalize(image, image_mean_, image_std_);
    }

    processed_images.push_back(image);
  }

  return torch::stack(processed_images);
}

std::vector<int64_t> CLIPImageProcessor::get_resize_output_image_size(
    const torch::Tensor& image,
    int shortest_edge) {
  int height = image.size(1);
  int width = image.size(2);

  int short_size = std::min(height, width);
  int long_size = std::max(height, width);

  int new_short = shortest_edge;
  int new_long = static_cast<int>(shortest_edge *
                                  static_cast<float>(long_size) / short_size);

  return height < width ? std::vector<int64_t>({new_short, new_long})
                        : std::vector<int64_t>({new_long, new_short});
}

}  // namespace xllm
