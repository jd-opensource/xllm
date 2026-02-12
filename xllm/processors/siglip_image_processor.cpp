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

#include "siglip_image_processor.h"

namespace xllm {

SiglipImageProcessor::SiglipImageProcessor(const ModelArgs& args) {
  do_resize_ = args.mm_image_do_resize();
  do_center_crop_ = args.mm_image_do_center_crop();
  do_rescale_ = args.mm_image_do_rescale();
  do_normalize_ = args.mm_image_do_normalize();
  crop_size_ = std::make_pair(args.mm_image_crop_height_size(),
                              args.mm_image_crop_width_size());
  resample_ = args.mm_image_resample();
  rescale_factor_ = args.mm_image_rescale_factor();
  image_mean_ = args.mm_image_normalize_mean();
  image_std_ = args.mm_image_normalize_std();
  size_ = {args.mm_image_size_height(), args.mm_image_size_width()};
}

bool SiglipImageProcessor::process(const MMInput& mm_inputs, MMData& mm_datas) {
  return false;
}

torch::Tensor SiglipImageProcessor::preprocess(const torch::Tensor& images) {
  int64_t batch_size = images.size(0);
  std::vector<torch::Tensor> processed_images;

  for (int64_t i = 0; i < batch_size; ++i) {
    torch::Tensor image = images[i];

    if (do_resize_) {
      image = resize(image, size_, resample_);
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

}  // namespace xllm
