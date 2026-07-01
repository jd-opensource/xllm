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

#include "musa_tvmffi_stream.h"

#include <glog/logging.h>
#include <tvm/ffi/extra/c_env_api.h>

#if defined(XLLM_TORCH_MUSA)
// torch_musa ships `c10::musa::*` symbols in two parallel header trees: the
// legacy `torch_musa/csrc/core/*.h` and the CUDA-compat shim `c10/musa/*.h`
// generated from PyTorch's `c10/cuda/*.h`. Including both in the same TU
// causes redefinition (MUSAStream, CaptureStatus, etc.) because both trees
// define the same classes in the same namespace. The musamapping plugin's
// `torch-include.json` path-rewrites `c10/cuda/*.h -> torch_musa/csrc/core/*`,
// so other TUs in xllm transitively pull in the csrc/core path. Use csrc/core
// directly here for consistency.
#include <torch_musa/csrc/core/MUSAGraphsC10Utils.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <array>
#include <optional>

namespace xllm::kernel::cuda {
namespace {

c10::musa::MUSAStream& get_or_create_tvmffi_musa_stream(
    c10::DeviceIndex device_index) {
  static thread_local std::array<std::optional<c10::musa::MUSAStream>, 8> slots;
  TORCH_CHECK(device_index >= 0 &&
                  device_index < static_cast<c10::DeviceIndex>(slots.size()),
              "invalid MUSA device index: ", device_index);
  std::optional<c10::musa::MUSAStream>& slot =
      slots[static_cast<size_t>(device_index)];
  if (!slot.has_value()) {
    slot = c10::musa::getStreamFromPool(/*isHighPriority=*/false, device_index);
  }
  return slot.value();
}

void set_tvmffi_stream_handle(c10::DeviceIndex device_index, void* stream) {
  constexpr int32_t kDlCuda = 2;
  constexpr int32_t kDlExtDev = 12;
  for (const int32_t device_type : {kDlCuda, kDlExtDev}) {
    void* original_stream = nullptr;
    const int rc = TVMFFIEnvSetStream(device_type, device_index, stream,
                                      &original_stream);
    if (rc != 0) {
      LOG(WARNING) << "[tvmffi.stream] failed to set stream, rc=" << rc
                   << " dev_type=" << device_type << " dev=" << device_index;
    }
  }
}

// True iff the currently-bound MUSA stream is in active graph capture. Any
// host-side sync (or stream allocation) is illegal in this state on
// torch_musa 2.7.1 -- musaStreamSynchronize / musaStreamWaitEvent /
// musaStreamCreate all fail with "operation not permitted when stream is
// capturing". Mirrors PyTorch's at::cuda::currentStreamCaptureStatus check
// used by FlashInfer for the same purpose.
bool is_current_stream_capturing() {
  return c10::musa::currentStreamCaptureStatusMayInitCtx() !=
         c10::musa::CaptureStatus::None;
}

}  // namespace

void bind_musa_tvmffi_stream(const torch::Device& device) {
  if (!is_torch_musa_device(device)) {
    return;
  }
  c10::musa::MUSAGuard device_guard(device.index());
  // Under graph capture, the Mate / TVM-FFI run must execute on the *current*
  // (captured) MUSA stream so its kernels are recorded into the graph. Using
  // the pooled non-default stream here would silently drop the FFI work out
  // of the captured DAG -- the graph would replay attention as a no-op.
  // Outside capture, prefer the pooled stream so FFI work can overlap with
  // torch compute on the default stream (required for the S5000 perf path).
  c10::musa::MUSAStream musa_stream =
      is_current_stream_capturing()
          ? c10::musa::getCurrentMUSAStream(device.index())
          : get_or_create_tvmffi_musa_stream(device.index());
  void* const stream = reinterpret_cast<void*>(musa_stream.stream());
  if (stream == nullptr) {
    LOG(ERROR) << "[tvmffi.stream] MUSA stream handle is null on " << device;
    return;
  }
  set_tvmffi_stream_handle(device.index(), stream);
}

void sync_current_musa_stream(const torch::Device& device) {
  if (!is_torch_musa_device(device)) {
    return;
  }
  // The captured graph encodes its own intra-graph dependencies; host-side
  // stream syncs are illegal here ("operation not permitted when stream is
  // capturing"). The post-replay barrier is handled by the graph executor.
  if (is_current_stream_capturing()) {
    return;
  }
  c10::musa::MUSAGuard device_guard(device.index());
  c10::musa::getCurrentMUSAStream(device.index()).synchronize();
}

void sync_musa_ffi_stream(const torch::Device& device) {
  if (!is_torch_musa_device(device)) {
    return;
  }
  // Same rationale as sync_current_musa_stream: skip during capture. The FFI
  // ran on the captured stream itself (see bind_musa_tvmffi_stream above), so
  // there is no cross-stream barrier to flush.
  if (is_current_stream_capturing()) {
    return;
  }
  c10::musa::MUSAGuard device_guard(device.index());
  get_or_create_tvmffi_musa_stream(device.index()).synchronize();
}

MusaTvmffiStreamGuard::MusaTvmffiStreamGuard(const torch::Device& device)
    : device_(device), active_(is_torch_musa_device(device)) {
  if (!active_) {
    return;
  }
  sync_current_musa_stream(device_);
  bind_musa_tvmffi_stream(device_);
}

MusaTvmffiStreamGuard::~MusaTvmffiStreamGuard() {
  if (!active_) {
    return;
  }
  sync_musa_ffi_stream(device_);
}

}  // namespace xllm::kernel::cuda

#else  // defined(XLLM_TORCH_MUSA)

namespace xllm::kernel::cuda {

void bind_musa_tvmffi_stream(const torch::Device& /*device*/) {}

void sync_current_musa_stream(const torch::Device& /*device*/) {}

void sync_musa_ffi_stream(const torch::Device& /*device*/) {}

MusaTvmffiStreamGuard::MusaTvmffiStreamGuard(const torch::Device& /*device*/) {}

MusaTvmffiStreamGuard::~MusaTvmffiStreamGuard() {}

}  // namespace xllm::kernel::cuda

#endif  // defined(XLLM_TORCH_MUSA)
