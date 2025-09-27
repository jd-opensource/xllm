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

#include "xtensor.h"

#if defined(USE_NPU)
#include "acl/acl.h"
#endif
#include "common/global_flags.h"

namespace xllm {
XTensor::XTensor(const Options& options, torch::ScalarType dtype)
    : options_(options), dtype_(dtype) {
  cache_size_per_token_ = options_.num_kv_heads() * options_.head_size() *
                          torch::scalarTypeToTypeMeta(dtype_).itemsize();

  buffer_size_per_seq_ = cache_size_per_token_ * options_.max_context_len();

  // align up to granularity size
  int64_t granularity_size = FLAGS_granularity_size;
  buffer_size_per_seq_ = (buffer_size_per_seq_ + granularity_size - 1) /
                         granularity_size * granularity_size;
  FLAGS_buffer_size_per_seq = buffer_size_per_seq_;

  // buffer size for all sequences
  buffer_size_ = buffer_size_per_seq_ * options_.max_seqs_per_batch();

  reserve_base_ptr();
}

XTensor::XTensor(int64_t buffer_size) : buffer_size_(buffer_size) {
  options_.max_seqs_per_batch() = 1;
  reserve_base_ptr();
}

XTensor::~XTensor() {
#if defined(USE_NPU)
  VmmResult status = aclrtReleaseMemAddress(base_ptr_);
  CHECK_EQ(status, VmmSuccess) << "Failed to free virtual memory for xtensor";
#endif
}

void XTensor::reserve_base_ptr() {
#if defined(USE_NPU)
  VmmResult status =
      aclrtReserveMemAddress(&base_ptr_, buffer_size_, 0, nullptr, 0);
  CHECK_EQ(status, VmmSuccess)
      << "Failed to reserve virtual memory for xtensor";
#endif
}
}  // namespace xllm
