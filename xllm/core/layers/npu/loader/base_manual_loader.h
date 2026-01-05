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

#include "base_loader.h"

namespace xllm {
namespace layer {

class BaseManualLoader : public BaseLoader {
 public:
  BaseManualLoader(uint64_t weight_count, const ModelContext& context);

  virtual ~BaseManualLoader() override;

  virtual void copy_weights_to_pinned_host();

  virtual void copy_weights_to_device();

  virtual void copy_weights_to_device_async();

  virtual void init_device_at_weights();

  virtual void init_weight_slices();

 protected:
  struct WeightSlice {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype = torch::kFloat16;
  };
  void* host_pinned_storage_ = nullptr;
  void* device_storage_ = nullptr;
  uint64_t storage_size_ = 0;
  std::vector<WeightSlice> weight_slices_;
  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;

  virtual bool is_nz_format_tensor(int weight_index) { return false; };
  void release_device_storage();
  void release_host_storage();
  int copy_host_nd_to_nz(torch::Tensor host_tensor,
                         void* dst_ptr,
                         uint64_t len);
  torch::Tensor convert_to_torch_tensor(const std::vector<int64_t>& dims,
                                        const torch::ScalarType dtype,
                                        const uintptr_t& dev_addr,
                                        int acl_format = 2);
};

}  // namespace layer
}  // namespace xllm