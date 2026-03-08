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
#include "rolling_weight_buffer.h"

namespace xllm {
namespace layer {

class BaseManualLoader : public BaseLoader {
 public:
  BaseManualLoader(uint64_t weight_count, const ModelContext& context);

  virtual ~BaseManualLoader() override;

  virtual void copy_weights_to_pinned_host();

  virtual void copy_weights_to_device();

  virtual void copy_weights_to_device_async();

  // Async H2D using the specified ACL stream (used by RollingLoadManager).
  virtual void copy_weights_to_device_async(aclrtStream stream);

  virtual void init_device_at_weights();

  virtual void init_weight_slices();

  virtual void merge_and_move_pinned_host() override;

  virtual void merge_loaded_weights() override;

  virtual void free_weights() override;

  virtual void reload_weights() override;

  virtual void reload_weights_from_device() override;

  // Rolling load path: refresh device slot pointer from rolling buffer and
  // rebuild AT tensor views from latest device base.
  virtual void refresh_rolling_weights() override;

  // Rolling load support: set the shared rolling buffer and this layer's index.
  // Device slot pointer / AT tensor views are refreshed via
  // refresh_rolling_weights().
  void set_rolling_buffer(std::shared_ptr<RollingWeightBuffer> buf,
                          int32_t layer_index);
  void* get_host_pinned_storage() const { return host_pinned_storage_; }
  uint64_t get_storage_size() const { return storage_size_; }

  // Allocate device storage (public for rolling load manager usage).
  void allocate_device_storage();

 protected:
  struct WeightSlice {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype = torch::kFloat16;
  };

  virtual void merge_host_at_weights() = 0;
  std::string model_id_;
  void* host_pinned_storage_ = nullptr;
  void* device_storage_ = nullptr;
  uint64_t storage_size_ = 0;
  std::vector<WeightSlice> weight_slices_;
  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;

  virtual bool is_nz_format_tensor(int weight_index) { return false; };

  void release_device_storage();
  void release_host_storage();

  std::shared_ptr<RollingWeightBuffer> rolling_buffer_ = nullptr;
  int32_t layer_index_ = -1;
  int copy_host_nd_to_nz(torch::Tensor host_tensor,
                         void* dst_ptr,
                         uint64_t len,
                         aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE);
  torch::Tensor convert_to_torch_tensor(const std::vector<int64_t>& dims,
                                        const torch::ScalarType dtype,
                                        const uintptr_t& dev_addr,
                                        int acl_format = ACL_FORMAT_ND);
};

}  // namespace layer
}  // namespace xllm
