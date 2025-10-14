#include "stream.h"

namespace xllm {

#if defined(USE_NPU)
Stream::Stream() : stream_(c10_npu::getNPUStreamFromPool()) {}
#elif defined(USE_MLU)
// TODO(mlu): implement mlu create stream
#endif

int Stream::synchronize() const {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(stream_.stream());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
}

c10::StreamGuard Stream::set_stream_guard() const {
  return c10::StreamGuard(stream_.unwrap());
}

}  // namespace xllm