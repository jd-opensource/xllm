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

#include <vector>

#include "core/framework/request/mm_input_helper.h"
#include "image_processor.h"

namespace xllm {

struct MMData;

class PyWarpperImageProcessor : public ImageProcessor {
 public:
  PyWarpperImageProcessor(const ModelArgs&);
  ~PyWarpperImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
};

}  // namespace xllm
