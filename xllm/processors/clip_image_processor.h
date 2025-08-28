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

#include <vector>

#include "core/util/tensor_helper.h"
#include "image_processor.h"

namespace xllm {

class CLIPImageProcessor : public ImageProcessor {
 public:
  CLIPImageProcessor(const ModelArgs& args);
  ~CLIPImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
  torch::Tensor process_images(const torch::Tensor& images);

 private:
  std::vector<int64_t> get_resize_output_image_size(const torch::Tensor& image,
                                                    int shortest_edge);

 private:
  bool do_resize_;
  bool do_center_crop_;
  bool do_rescale_;
  bool do_normalize_;
  int shortest_edge_;
  int resample_;
  double rescale_factor_;
  std::pair<int, int> crop_size_;
  std::vector<double> image_mean_;
  std::vector<double> image_std_;
};

}  // namespace xllm
