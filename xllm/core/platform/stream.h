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

class Stream {
 public:
  Stream();
  ~Stream() = default;

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&&) = default;
  Stream& operator=(Stream&&) = default;

  int synchronize_stream() const;
  c10::StreamGuard set_stream_guard() const;

 private:
#if defined(USE_NPU)
  c10_npu::NPUStream stream_;
#elif defined(USE_MLU)
// TODO(mlu): implement mlu stream
#endif
};

}  // namespace xllm