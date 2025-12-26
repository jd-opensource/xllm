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

inline bool safe_concat(const std::vector<torch::Tensor>& vec,
                        torch::Tensor& tar,
                        int64_t dim = 0) {
  auto check = [](const std::vector<torch::Tensor>& vec, int64_t dim) {
    if (vec.empty()) return false;

    const auto& ref = vec[0];
    if (!ref.defined()) return false;

    const int64_t ndim = ref.dim();
    if (ndim == 0) return false;

    if (dim < 0) dim += ndim;

    if (dim >= ndim) return false;

    for (size_t i = 1; i < vec.size(); ++i) {
      const auto& t = vec[i];
      if (!t.defined()) {
        return false;
      }

      if (t.dtype() != ref.dtype() || t.device() != ref.device() ||
          t.dim() != ref.dim()) {
        return false;
      }

      for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (t.size(d) != ref.size(d)) {
          return false;
        }
      }
    }
    return true;
  };

  if (check(vec, dim)) {
    tar = torch::cat(vec, dim);
    return true;
  } else {
    return false;
  }
}

// save torch tensor to .pt file as pickle format, which is same as torch.save
// in python. .pt file can be loaded by torch.load in python. file_path must end
// with ".pt".
inline void save_tensor_as_pickle(const torch::Tensor& tensor,
                                  const std::string& file_path) {
  std::vector<char> pickled = torch::pickle_save(tensor);
  std::ofstream ofs(file_path, std::ios::binary);
  CHECK(ofs.good()) << "Cannot open file: " << file_path;
  ofs.write(pickled.data(), static_cast<std::streamsize>(pickled.size()));
  CHECK(ofs.good()) << "Write failed to: " << file_path;
}

}  // namespace xllm
