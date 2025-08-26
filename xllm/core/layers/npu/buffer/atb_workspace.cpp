
#include "atb_workspace.h"

#include <acl/acl.h>

#include "atb_buffer.h"
#include "common/device_monitor.h"

namespace xllm {

std::map<int32_t, std::unique_ptr<AtbBuffer>> AtbWorkspace::buffer_map_;

AtbWorkspace::AtbWorkspace(at::Device device) {
  int32_t device_id = device.index();
  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    buffer_map_[device_id] = std::make_unique<AtbBuffer>(1, device);
  }
}

AtbWorkspace::~AtbWorkspace() {}

void* AtbWorkspace::get_workspace_buffer(uint64_t bufferSize) {
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);

  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    LOG(FATAL) << " Fail to find device_id in buffer_map_ : " << device_id;
  }
  DeviceMonitor::get_instance().monitor_buffer(device_id, bufferSize);

  return it->second->get_buffer(bufferSize);
}

}  // namespace xllm