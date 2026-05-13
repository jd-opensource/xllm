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

#include "platform/mlu/mlu_tensor_alloc.h"

#include <cnrt.h>
#include <glog/logging.h>

#include <limits>

namespace xllm::mlu {

namespace {

size_t get_nbytes(const std::vector<int64_t>& dims,
                  const torch::ScalarType dtype) {
  size_t count = 1;
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "tensor dim must be non-negative";
    const size_t dim_size = static_cast<size_t>(dim);
    if (dim_size > static_cast<size_t>(0)) {
      CHECK_LE(count, std::numeric_limits<size_t>::max() / dim_size)
          << "tensor element count overflow";
    }
    count *= dim_size;
  }
  const size_t elem_size = static_cast<size_t>(torch::elementSize(dtype));
  CHECK_GT(elem_size, static_cast<size_t>(0)) << "tensor dtype size is zero";
  CHECK_LE(count, std::numeric_limits<size_t>::max() / elem_size)
      << "tensor byte size overflow";
  return count * elem_size;
}

void free_tensor(void* ptr, int32_t device_id) {
  if (ptr == nullptr) {
    return;
  }

  cnrtRet_t ret = cnrtSetDevice(device_id);
  CHECK(ret == cnrtSuccess)
      << "cnrtSetDevice failed, ret=" << static_cast<int32_t>(ret)
      << ", device_id=" << device_id;
  ret = cnrtFree(ptr);
  CHECK(ret == cnrtSuccess)
      << "cnrtFree failed, ret=" << static_cast<int32_t>(ret)
      << ", ptr=" << ptr;
}

}  // namespace

torch::Tensor alloc_zero_tensor(const std::vector<int64_t>& dims,
                                torch::ScalarType dtype,
                                const torch::Device& device) {
  CHECK(device.has_index()) << "MLU device index is required";
  int32_t device_id = static_cast<int32_t>(device.index());

  cnrtRet_t ret = cnrtSetDevice(device_id);
  CHECK(ret == cnrtSuccess)
      << "cnrtSetDevice failed, ret=" << static_cast<int32_t>(ret)
      << ", device_id=" << device_id;

  size_t nbytes = get_nbytes(dims, dtype);
  void* ptr = nullptr;
  ret = cnrtMalloc(&ptr, nbytes);
  CHECK(ret == cnrtSuccess)
      << "cnrtMalloc failed, ret=" << static_cast<int32_t>(ret)
      << ", nbytes=" << nbytes;
  ret = cnrtMemset(ptr, 0, nbytes);
  CHECK(ret == cnrtSuccess)
      << "cnrtMemset failed, ret=" << static_cast<int32_t>(ret)
      << ", nbytes=" << nbytes;

  auto deleter = [device_id](void* data) { free_tensor(data, device_id); };
  auto options =
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
  return torch::from_blob(ptr, dims, deleter, options);
}

}  // namespace xllm::mlu
