#include "device_monitor_async_wrapper.h"

namespace xllm {

folly::SemiFuture<const DeviceStats*>
DeviceMonitorAsyncWrapper::get_device_stats_async(int32_t device_id) {
  folly::Promise<const DeviceStats*> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, device_id, promise = std::move(promise)]() mutable {
        const DeviceStats& device_stats =
            device_monitor_->get_device_stats(device_id);
        promise.setValue(&device_stats);
      });
  return future;
}

}  // namespace xllm