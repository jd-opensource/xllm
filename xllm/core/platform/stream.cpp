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
Stream::Stream() : stream_(c10_npu::getNPUStreamFromPool()) {}
#elif defined(USE_MLU)
Stream::Stream() : stream_(torch_mlu::getStreamFromPool()) {}
#endif

int Stream::synchronize() const {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(stream_.stream());
#elif defined(USE_MLU)
  stream_.unwrap().synchronize();
  return 0;
#endif
}

c10::StreamGuard Stream::set_stream_guard() const {
  return c10::StreamGuard(stream_.unwrap());
}

}  // namespace xllm
