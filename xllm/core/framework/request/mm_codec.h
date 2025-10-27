/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <opencv2/opencv.hpp>
#include <string>

namespace xllm {

class OpenCVImageDecoder {
 public:
  OpenCVImageDecoder() = default;
  ~OpenCVImageDecoder() = default;

  bool decode(const std::string& raw_data, torch::Tensor& t);
};

class OpenCVImageEncoder {
 public:
  OpenCVImageEncoder() = default;
  ~OpenCVImageEncoder() = default;

  bool encode(const torch::Tensor& t, std::string& raw_data);

 private:
  bool valid(const torch::Tensor& t);
};

}  // namespace xllm
