#include "stream_helper.h"

namespace xllm {

void synchronize_stream(int32_t device_id) {
#if defined(USE_NPU)
  aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(device_id).stream());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
}

#if defined(USE_NPU)
StreamHelper::StreamHelper() : stream_(c10_npu::getNPUStreamFromPool()) {}
#elif defined(USE_MLU)
// TODO(mlu): implement mlu create stream
#endif

void StreamHelper::synchronize_stream() {
#if defined(USE_NPU)
  aclrtSynchronizeStream(stream_.stream());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
}

int StreamHelper::synchronize_stream_return_status() {
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