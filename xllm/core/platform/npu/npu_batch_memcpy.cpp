/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstring>

namespace xllm {
namespace npu {

void NPUBatchMemcpy::init(int32_t device_id) {
  if (initialized_) {
    return;
  }
  device_id_ = device_id;

  memset(&h2d_attr_, 0, sizeof(h2d_attr_));
  h2d_attr_.dstLoc.id = device_id;
  h2d_attr_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
  h2d_attr_.srcLoc.id = device_id;
  h2d_attr_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;

  memset(&d2h_attr_, 0, sizeof(d2h_attr_));
  d2h_attr_.dstLoc.id = device_id;
  d2h_attr_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
  d2h_attr_.srcLoc.id = device_id;
  d2h_attr_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;

  initialized_ = true;
}

bool NPUBatchMemcpy::copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(src_tensors, dst_tensors, h2d_attr_, stream);
}

bool NPUBatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(src_tensors, dst_tensors, d2h_attr_, stream);
}

bool NPUBatchMemcpy::copy(const std::vector<torch::Tensor>& src_tensors,
                          const std::vector<torch::Tensor>& dst_tensors,
                          const aclrtMemcpyBatchAttr& attr,
                          Stream* stream) {
  CHECK(initialized_) << "NPUBatchMemcpy not initialized.";
  CHECK(stream != nullptr) << "Stream must not be null.";
  CHECK_EQ(src_tensors.size(), dst_tensors.size())
      << "src and dst tensor count mismatch.";

  if (src_tensors.empty()) {
    return true;
  }

  const size_t count = src_tensors.size();
  CHECK_LE(count, kMaxBatchCopyCount)
      << "Batch copy count exceeds limit: " << count;

  std::vector<void*> srcs(count);
  std::vector<void*> dsts(count);
  std::vector<size_t> sizes(count);

  for (size_t i = 0; i < count; ++i) {
    srcs[i] = src_tensors[i].data_ptr();
    dsts[i] = dst_tensors[i].data_ptr();
    sizes[i] = static_cast<size_t>(src_tensors[i].nbytes());
  }

  c10::StreamGuard guard = stream->set_stream_guard();

  aclrtMemcpyBatchAttr attrs[1] = {attr};
  size_t attrs_indexes[1] = {0};
  size_t fail_index = SIZE_MAX;

  aclError ret = aclrtMemcpyBatch(dsts.data(),
                                  sizes.data(),
                                  srcs.data(),
                                  sizes.data(),
                                  count,
                                  attrs,
                                  attrs_indexes,
                                  1,
                                  &fail_index);
  if (ret != ACL_SUCCESS || fail_index != SIZE_MAX) {
    LOG(ERROR) << "aclrtMemcpyBatch failed: ret=" << ret
               << ", fail_index=" << fail_index;
    return false;
  }

  if (stream->synchronize() != 0) {
    LOG(ERROR) << "BatchMemcpy stream synchronize timeout.";
    return false;
  }

  return true;
}

}  // namespace npu
}  // namespace xllm
