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

#include <cstdint>
#include <vector>

#include "common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "framework/xtensor/xtensor.h"

namespace xllm {
class KVCache final {
 public:
  KVCache() = default;
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);
  KVCache(torch::Tensor key_cache,
          torch::Tensor value_cache,
          torch::Tensor index_cache);
  KVCache(std::shared_ptr<XTensor> key_xtensor,
          std::shared_ptr<XTensor> value_xtensor);
  ~KVCache() = default;

  // TODO: pass in kv_shape and options instead
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;
  torch::Tensor get_index_cache() const;

  std::vector<std::vector<int64_t>> get_shapes();

  std::shared_ptr<XTensor> get_k_xtensor() const;
  std::shared_ptr<XTensor> get_v_xtensor() const;

  bool empty() const {
    return FLAGS_enable_continuous_kvcache
               ? (key_xtensor_ == nullptr || value_xtensor_ == nullptr)
               : (!key_cache_.defined() || !value_cache_.defined());
  }

  void swap_blocks(torch::Tensor& src_tensor, torch::Tensor& dst_tensor);

 private:
  torch::Tensor key_cache_;
  torch::Tensor value_cache_;
  torch::Tensor index_cache_;

  // for continuous kvcache
  std::shared_ptr<XTensor> key_xtensor_;
  std::shared_ptr<XTensor> value_xtensor_;
};

}  // namespace xllm
