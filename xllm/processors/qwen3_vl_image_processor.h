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

#include "qwen2_vl_image_processor.h"

namespace xllm {

class Qwen3VLImageProcessor : public Qwen2VLImageProcessor {
 public:
  explicit Qwen3VLImageProcessor(const ModelArgs& args)
      : Qwen2VLImageProcessor(args) {}

  std::optional<Size> smart_resize_video(int num_frames,
                                         int height,
                                         int width,
                                         int temporal_factor,
                                         int factor,
                                         int min_pixels,
                                         int max_pixels) const override;

  torch::Tensor sample_frames(const VideoMetadata& metadata,
                              int temporal_patch_size,
                              int min_frames,
                              int max_frames,
                              int num_frames,
                              double set_fps) override;
};

}  // namespace xllm
