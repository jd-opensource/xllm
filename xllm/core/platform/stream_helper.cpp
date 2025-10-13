#include "stream_helper.h"

namespace xllm {

#if defined(USE_NPU)
StreamHelper::StreamHelper() : stream_(c10_npu::getNPUStreamFromPool()) {}
#elif defined(USE_MLU)
// TODO(mlu): implement mlu create stream
#endif

int StreamHelper::synchronize_stream() {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(stream_.stream());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
}

c10::StreamGuard StreamHelper::set_stream_guard() {
  return c10::StreamGuard(stream_.unwrap());
}

}  // namespace xllm