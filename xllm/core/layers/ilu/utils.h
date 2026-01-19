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

inline void save_tensor(const torch::Tensor& tensor, const std::string& name) {
  if (!tensor.defined()) {
    std::cout << "[Warning] Skip saving " << name << ": Tensor is undefined."
              << std::endl;
    return;
  }
  std::string root = "/workspace/tmp/";
  torch::Tensor tensor_tmp = tensor.cpu().contiguous();

  auto shape = tensor_tmp.sizes();
  auto dtype = tensor_tmp.dtype();

  std::ostringstream filename;
  filename << name << "_shape[";
  for (size_t i = 0; i < shape.size(); ++i) {
    filename << shape[i];
    if (i < shape.size() - 1) {
      filename << ",";
    }
  }
  filename << "]_dtype[" << dtype << "].pt";
  std::string save_path = root + filename.str();
  torch::save(tensor_tmp, save_path);
  std::cout << "successfully save " << save_path << std::endl;
}