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

#include <vector>

#include "common/macros.h"
#include "util/type_traits.h"

namespace xllm {
// for all sequences
// for one layer
// k or v cache
class XTensor final {
 public:
  struct Options {
    PROPERTY(int64_t, num_kv_heads) = 0;
    PROPERTY(int64_t, head_size) = 0;
    PROPERTY(int32_t, max_context_len) = 0;
    PROPERTY(int32_t, max_seqs_per_batch) = 0;
  };

  XTensor(const Options& options, torch::ScalarType dtype);
  XTensor(int64_t buffer_size);

  XTensor(XTensor&&) = default;
  XTensor& operator=(XTensor&&) = default;
  XTensor(const XTensor&) = delete;
  XTensor& operator=(const XTensor&) = delete;

  ~XTensor();

  VirPtr get_base_ptr() const {
    CHECK(base_ptr_ != nullptr) << "Base pointer is not initialized";
    return base_ptr_;
  }

  VirPtr get_vir_ptr(int32_t seq_id) const {
    CHECK(base_ptr_ != nullptr) << "Base pointer is not initialized";
    return reinterpret_cast<VirPtr>((char*)base_ptr_ +
                                    seq_id * buffer_size_per_seq_);
  }

  const Options& options() const { return options_; }

 private:
  void reserve_base_ptr();

 private:
  Options options_;
  int64_t buffer_size_;
  torch::ScalarType dtype_;
  int64_t buffer_size_per_seq_;
  int64_t cache_size_per_token_;
  // the start virtual pointer of the xtensor
  VirPtr base_ptr_ = nullptr;
};
}  // namespace xllm