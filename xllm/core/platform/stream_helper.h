#pragma once

// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
// clang-format on

#include <cstdint>
#if defined(USE_NPU)
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>
#endif

namespace xllm {

void synchronize_stream(int32_t device_id);

class StreamHelper {
 public:
  StreamHelper();
  ~StreamHelper() = default;

  StreamHelper(const StreamHelper&) = delete;
  StreamHelper& operator=(const StreamHelper&) = delete;
  StreamHelper(StreamHelper&&) = default;
  StreamHelper& operator=(StreamHelper&&) = default;

  void synchronize_stream();
  int synchronize_stream_return_status();
  c10::StreamGuard set_stream_guard();

 private:
#if defined(USE_NPU)
  c10_npu::NPUStream stream_;
#elif defined(USE_MLU)
// TODO(mlu): implement mlu stream
#endif
};

}  // namespace xllm