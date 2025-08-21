#pragma once

#include <folly/futures/Future.h>

#include "device_monitor.h"
#include "util/threadpool.h"

namespace xllm {

class DeviceMonitorAsyncWrapper {
 public:
  explicit DeviceMonitorAsyncWrapper() {
    device_monitor_ = &DeviceMonitor::get_instance();
  }

  folly::SemiFuture<const DeviceStats*> get_device_stats_async(
      int32_t device_id);

 private:
  DeviceMonitor* device_monitor_;

  ThreadPool threadpool_;
};
}  // namespace xllm