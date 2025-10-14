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
