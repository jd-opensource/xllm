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

#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/util/tensor_helper.h"
#include "image_processor.h"

namespace xllm {

class MiniCPMVImageProcessor : public ImageProcessor {
 public:
  MiniCPMVImageProcessor(const std::shared_ptr<ModelArgs>& args);
  ~MiniCPMVImageProcessor() override = default;

  static std::pair<int, int> get_sliced_grid(
      const std::pair<int, int>& original_size,
      int max_slice_nums,
      int scale_resolution,
      bool never_split = false);

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     std::vector<torch::Tensor>& new_images,
                     std::vector<torch::Tensor>& tgt_sizes);

  int ensure_divide(int length, int patch_size) const {
    return std::max(
        std::lround(static_cast<float>(length) / patch_size) * patch_size,
        static_cast<long>(patch_size));
  }

  std::pair<int, int> find_best_resize(const std::pair<int, int>& original_size,
                                       int scale_resolution,
                                       int patch_size,
                                       bool allow_upscale = false) const;

  std::pair<int, int> get_refine_size(const std::pair<int, int>& original_size,
                                      const std::pair<int, int>& grid,
                                      int scale_resolution,
                                      int patch_size,
                                      bool allow_upscale = false) const;

  std::tuple<torch::Tensor,
             std::vector<std::vector<torch::Tensor>>,
             std::pair<int, int>>
  slice_image(const torch::Tensor& image,
              int max_slice_nums = 9,
              int scale_resolution = 448,
              int patch_size = 14,
              bool never_split = false);

  std::vector<std::vector<torch::Tensor>> split_to_patches(
      const torch::Tensor& image,
      const std::pair<int, int>& grid) const;

  torch::Tensor reshape_by_patch(const torch::Tensor& image);

  std::vector<torch::Tensor> get_sliced_images(const torch::Tensor& image,
                                               int max_slice_nums = -1);

 private:
  bool slice_mode_;
  int max_slice_nums_;
  int scale_resolution_;
  int patch_size_;
  int image_feature_size_;
  std::vector<double> norm_mean_;
  std::vector<double> norm_std_;
};

}  // namespace xllm
