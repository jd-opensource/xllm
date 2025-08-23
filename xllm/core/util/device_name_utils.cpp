#include "device_name_utils.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#if defined(USE_NPU)
#include <c10/core/StorageImpl.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/torch_npu.h>
#elif defined(USE_MLU)
// TODO(mlu): include mlu device name utils
#endif

namespace xllm {

#if defined(USE_NPU)
std::vector<torch::Device> DeviceNameUtils::parse_devices(
    const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto" || device_str.empty()) {
    // use all available npus if any
    const auto num_npus = c10_npu::device_count();
    if (num_npus == 0) {
      LOG(INFO) << "no npus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_npus);
    for (int i = 0; i < num_npus; ++i) {
      std::string device_name = "npu:" + std::to_string(i);
      devices.emplace_back(torch::Device(device_name));
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    std::vector<std::string> parts = absl::StrSplit(device_str, ':');
    CHECK(parts.size() == 2) << "Invalid device string format: " << device_str;
    CHECK(parts[0] == "npu") << "Unsupported device type: " << parts[0];

    int device_index;
    CHECK(absl::SimpleAtoi(parts[1], &device_index))
        << "Invalid device index: " << parts[1];

    devices.emplace_back(c10::DeviceType::PrivateUse1, device_index);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << device_str;
  return devices;
}
#elif defined(USE_MLU)
// TODO(mlu): implement mlu device name utils
#endif

}  // namespace xllm
