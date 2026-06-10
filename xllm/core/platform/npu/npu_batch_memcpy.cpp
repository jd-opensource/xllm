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

#include "platform/npu/npu_batch_memcpy.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <memory>

namespace xllm {
namespace npu {

void NPUBatchMemcpy::init(int32_t device_id) {
  if (initialized_) {
    CHECK_EQ(device_id_, device_id)
        << "NPUBatchMemcpy already initialized with device_id=" << device_id_
        << ", but got device_id=" << device_id;
    return;
  }

  device_id_ = device_id;
  h2d_attr_ = build_h2d_attr(device_id);
  d2h_attr_ = build_d2h_attr(device_id);
  initialized_ = true;
}

bool NPUBatchMemcpy::copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  CHECK(stream != nullptr) << "stream must not be null.";
  CHECK(initialized_) << "NPUBatchMemcpy must be initialized before copy_h2d.";
  CHECK_EQ(stream->get_stream()->device_index(), device_id_)
      << "stream device_id does not match initialized device_id.";
  return copy(src_tensors, dst_tensors, *h2d_attr_, stream);
}

bool NPUBatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  CHECK(stream != nullptr) << "stream must not be null.";
  CHECK(initialized_) << "NPUBatchMemcpy must be initialized before copy_d2h.";
  CHECK_EQ(stream->get_stream()->device_index(), device_id_)
      << "stream device_id does not match initialized device_id.";
  return copy(src_tensors, dst_tensors, *d2h_attr_, stream);
}

bool NPUBatchMemcpy::copy(const std::vector<torch::Tensor>& src_tensors,
                          const std::vector<torch::Tensor>& dst_tensors,
                          const aclrtMemcpyBatchAttr& attr,
                          Stream* stream) {
  CHECK(stream != nullptr) << "stream must not be null.";
  CHECK_EQ(src_tensors.size(), dst_tensors.size())
      << "src_tensors.size() must equal dst_tensors.size().";

  if (src_tensors.empty()) {
    return true;
  }

  std::vector<void*> src_ptrs;
  std::vector<void*> dst_ptrs;
  std::vector<size_t> copy_sizes;
  src_ptrs.reserve(src_tensors.size());
  dst_ptrs.reserve(dst_tensors.size());
  copy_sizes.reserve(src_tensors.size());

  for (size_t i = 0; i < src_tensors.size(); ++i) {
    const auto& src_tensor = src_tensors[i];
    const auto& dst_tensor = dst_tensors[i];
    CHECK(src_tensor.defined())
        << "src tensor at index " << i << " is undefined.";
    CHECK(dst_tensor.defined())
        << "dst tensor at index " << i << " is undefined.";
    CHECK(src_tensor.is_contiguous())
        << "src tensor at index " << i << " must be contiguous.";
    CHECK(dst_tensor.is_contiguous())
        << "dst tensor at index " << i << " must be contiguous.";

    const size_t src_size =
        static_cast<size_t>(src_tensor.numel()) * src_tensor.element_size();
    const size_t dst_size =
        static_cast<size_t>(dst_tensor.numel()) * dst_tensor.element_size();
    CHECK_EQ(src_size, dst_size)
        << "src and dst tensor bytes mismatch at index " << i << ": "
        << src_size << " vs " << dst_size;

    src_ptrs.emplace_back(src_tensor.data_ptr());
    dst_ptrs.emplace_back(dst_tensor.data_ptr());
    copy_sizes.emplace_back(src_size);
  }

  c10::StreamGuard stream_guard = stream->set_stream_guard();
  aclrtMemcpyBatchAttr attrs[1] = {attr};
  size_t attrs_indexes[1] = {0};

  for (size_t begin = 0; begin < src_ptrs.size(); begin += kMaxBatchCopyCount) {
    const size_t end = std::min(begin + kMaxBatchCopyCount, src_ptrs.size());
    const size_t copy_count = end - begin;
    size_t fail_index = SIZE_MAX;
    aclError ret = aclrtMemcpyBatch(dst_ptrs.data() + begin,
                                    copy_sizes.data() + begin,
                                    src_ptrs.data() + begin,
                                    copy_sizes.data() + begin,
                                    copy_count,
                                    attrs,
                                    attrs_indexes,
                                    1,
                                    &fail_index);
    if (ret != ACL_SUCCESS || fail_index != SIZE_MAX) {
      LOG(ERROR) << "aclrtMemcpyBatch error: " << ret
                 << ", fail_index: " << fail_index << ", batch_begin: " << begin
                 << ", batch_size: " << copy_count;
      return false;
    }
  }

  if (stream->synchronize() != 0) {
    LOG(ERROR) << "NPU batch memcpy synchronize timeout.";
    return false;
  }

  return true;
}

aclrtMemcpyBatchAttr NPUBatchMemcpy::build_h2d_attr(int32_t device_id) const {
  aclrtMemcpyBatchAttr attr;
  attr.dstLoc.id = device_id;
  attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  attr.srcLoc.id = device_id;
  attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
  memset(attr.rsv, 0, sizeof(attr.rsv));
  return attr;
}

aclrtMemcpyBatchAttr NPUBatchMemcpy::build_d2h_attr(int32_t device_id) const {
  aclrtMemcpyBatchAttr attr;
  attr.dstLoc.id = device_id;
  attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
  attr.srcLoc.id = device_id;
  attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  memset(attr.rsv, 0, sizeof(attr.rsv));
  return attr;
}

}  // namespace npu
}  // namespace xllm
