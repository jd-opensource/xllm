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

#include <limits>
#include <vector>

namespace xllm {

class Sequence;
struct ModelArgs;

class MPositionHelper {
 public:
  MPositionHelper(Sequence& seq, const ModelArgs& args)
      : seq_(seq), args_(args) {}

  torch::Tensor get_positions();

 private:
  std::tuple<torch::Tensor, int> get_positions_p(
      torch::Tensor image_grid_thw,
      torch::Tensor video_grid_thw,
      torch::Tensor second_per_grid_ts);
  std::tuple<torch::Tensor, int> get_positions_glm(
      torch::Tensor image_grid_thw,
      torch::Tensor video_grid_thw);

  torch::Tensor get_positions_d();

 private:
  Sequence& seq_;
  const ModelArgs& args_;
};

}  // namespace xllm
