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

#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <fstream>
#include <vector>

namespace xllm {

template <typename T>
inline torch::Tensor create_2d_tensor(const std::vector<std::vector<T> >& vec,
                                      torch::ScalarType dtype) {
  if (vec.empty()) {
    return {};
  }
  // create tensor on cpu pinned memory here
  const size_t n_rows = vec.size();
  const size_t n_cols = vec[0].size();
  auto tensor =
      torch::empty({static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
                   torch::TensorOptions()
                       .dtype(dtype)
                       .device(torch::kCPU)
                       .pinned_memory(true));
  for (int64_t i = 0; i < n_rows; ++i) {
    CHECK_EQ(vec[i].size(), n_cols);
    tensor[i] = torch::tensor(vec[i],
                              torch::TensorOptions()
                                  .dtype(dtype)
                                  .device(torch::kCPU)
                                  .pinned_memory(true));
  }
  return tensor;
};

// 为空的2D vector提供特殊优化版本
template <typename T>
inline torch::Tensor create_2d_tensor_optimized(
    const std::vector<std::vector<T> >& vec,
    torch::ScalarType dtype) {
  if (vec.empty()) {
    return {};
  }

  const size_t n_rows = vec.size();
  const size_t n_cols = vec.empty() ? 0 : vec[0].size();

  // 对于全零矩阵的特殊优化
  bool all_zero = true;
  for (const auto& row : vec) {
    for (const auto& val : row) {
      if (val != T(0)) {
        all_zero = false;
        break;
      }
    }
    if (!all_zero) break;
  }

  if (all_zero) {
    // 直接创建零tensor，更高效
    return torch::zeros(
        {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
        torch::TensorOptions()
            .dtype(dtype)
            .device(torch::kCPU)
            .pinned_memory(true));
  }

  // 否则使用优化的内存复制方式
  auto tensor =
      torch::empty({static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
                   torch::TensorOptions()
                       .dtype(dtype)
                       .device(torch::kCPU)
                       .pinned_memory(true));

  // 优化：使用批量内存复制替代逐行torch::tensor创建
  T* tensor_data = tensor.data_ptr<T>();
  for (int64_t i = 0; i < n_rows; ++i) {
    CHECK_EQ(vec[i].size(), n_cols);
    // 直接复制内存，避免创建临时tensor
    std::memcpy(tensor_data + i * n_cols, vec[i].data(), n_cols * sizeof(T));
  }
  return tensor;
};

inline torch::Tensor safe_to(const torch::Tensor& t,
                             const torch::TensorOptions& options,
                             bool non_blocking = false) {
  return t.defined() ? t.to(options, non_blocking) : t;
};

inline std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}

inline torch::Tensor load_tensor(std::string filename) {
  std::vector<char> f = get_the_bytes(filename);
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor my_tensor = x.toTensor();
  return my_tensor;
}

inline void print_tensor(const torch::Tensor& tensor,
                         const std::string& tensor_name = "tensor",
                         int num = 10,
                         bool part = true,
                         bool print_value = true) {
  if (!tensor.defined()) {
    LOG(INFO) << tensor_name << ", Undefined tensor." << std::endl;
    return;
  }

  LOG(INFO) << "======================================" << std::endl;
  LOG(INFO) << tensor_name << ": " << tensor.sizes()
            << ", dtype: " << tensor.dtype() << ", device: " << tensor.device()
            << std::endl;

  if (!print_value) {
    return;
  }

  if (part) {
    const auto& flat_tensor = tensor.contiguous().view(-1);
    int max_elements = std::min(static_cast<int>(flat_tensor.size(0)), num);
    // const auto& front_elements = flat_tensor.slice(0, 0, max_elements);
    const auto& front_elements =
        flat_tensor.slice(0, 0, max_elements).to(torch::kCPU);
    LOG(INFO) << "First " << max_elements << " elements: \n"
              << front_elements << std::endl;

    int back_num = flat_tensor.size(0) > num ? num : flat_tensor.size(0);
    // const auto& back_elements = flat_tensor.slice(0, flat_tensor.size(0) -
    // back_num, flat_tensor.size(0));
    const auto& back_elements =
        flat_tensor
            .slice(0, flat_tensor.size(0) - back_num, flat_tensor.size(0))
            .to(torch::kCPU);
    LOG(INFO) << "Last " << back_num << " elements: \n"
              << back_elements << std::endl;
  } else {
    LOG(INFO) << "All: \n" << tensor.to(torch::kCPU) << std::endl;
  }
}

inline bool file_exists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

inline torch::Tensor safe_concat(const torch::Tensor& t1,
                                 const torch::Tensor& t2,
                                 const uint32_t dim) {
  if (t1.defined() && t2.defined()) {
    return torch::cat({t1, t2}, dim);
  } else if (!t1.defined()) {
    return t2;
  } else {
    return t1;
  }
}

}  // namespace xllm