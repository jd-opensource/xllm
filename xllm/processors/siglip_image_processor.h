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

#include <torch/torch.h>

#include <vector>

#include "core/util/tensor_helper.h"
#include "image_processor.h"

namespace xllm {

class SiglipImageProcessor : public ImageProcessor {
 public:
  SiglipImageProcessor(const ModelArgs& args);
  ~SiglipImageProcessor() override = default;
  SiglipImageProcessor() = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
  torch::Tensor preprocess(const torch::Tensor& images);

 private:
  bool do_resize_;
  bool do_center_crop_;
  bool do_rescale_;
  bool do_normalize_;
  int64_t resample_;
  double rescale_factor_;
  std::pair<int64_t, int64_t> crop_size_;
  std::vector<double> image_mean_;
  std::vector<double> image_std_;
  std::vector<int64_t> size_;
};

}  // namespace xllm
