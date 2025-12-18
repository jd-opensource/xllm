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

#include "stream.h"

namespace xllm {

#if defined(USE_NPU)
Stream::Stream(const int32_t timeout)
    : stream_(c10_npu::getNPUStreamFromPool()), timeout_(timeout) {}
#elif defined(USE_MLU)
Stream::Stream(const int32_t timeout)
    : stream_(torch_mlu::getStreamFromPool()), timeout_(timeout) {}
#elif defined(USE_CUDA) || defined(USE_ILU)
Stream::Stream(const int32_t timeout)
    : stream_(c10::cuda::getStreamFromPool()), timeout_(timeout) {}
#endif

#if defined(USE_NPU)
Stream::Stream(c10_npu::NPUStream stream, const int32_t timeout)
    : stream_(stream), timeout_(timeout) {}
#elif defined(USE_MLU)
Stream::Stream(torch_mlu::MLUStream stream, const int32_t timeout)
    : stream_(stream), timeout_(timeout) {}
#elif defined(USE_CUDA) || defined(USE_ILU)
Stream::Stream(c10::cuda::CUDAStream stream, const int32_t timeout)
    : stream_(stream), timeout_(timeout) {}
#endif

int Stream::synchronize() const {
#if defined(USE_NPU)
  return aclrtSynchronizeStreamWithTimeout(stream_.stream(), timeout_);
#elif defined(USE_MLU)
  stream_.unwrap().synchronize();
  return 0;
#elif defined(USE_CUDA) || defined(USE_ILU)
  stream_.synchronize();
  return 0;
#else
  LOG(FATAL)
      << "Not supported backend, currently we support 'npu', 'cuda', 'mlu'.";
#endif
}

c10::StreamGuard Stream::set_stream_guard() const {
#if defined(USE_CUDA) || defined(USE_ILU)
  return c10::StreamGuard(stream_);
#else
  return c10::StreamGuard(stream_.unwrap());
#endif
}

void Stream::wait_stream(const Stream& other_stream) {
  // get the c10::Stream objects for the current stream and the other stream
#if defined(USE_CUDA) || defined(USE_ILU)
  const c10::Stream& current_c10_stream = this->stream_;
  const c10::Stream& target_c10_stream = other_stream.stream_;
#else
  c10::Stream current_c10_stream = this->stream_.unwrap();
  c10::Stream target_c10_stream = other_stream.stream_.unwrap();
#endif

  c10::Event event(current_c10_stream.device_type());
  event.record(target_c10_stream);
  event.block(current_c10_stream);
}

}  // namespace xllm
