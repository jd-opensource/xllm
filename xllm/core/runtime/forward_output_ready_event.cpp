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

#include "runtime/forward_output_ready_event.h"

#include <glog/logging.h>

#include <mutex>

#include "platform/stream.h"

#if defined(USE_NPU)
#include <acl/acl_rt.h>
#endif

namespace xllm {

#if defined(USE_NPU)

namespace {

class NpuForwardOutputReadyEvent final : public ForwardOutputReadyEvent {
 public:
  explicit NpuForwardOutputReadyEvent(aclrtStream stream) {
    aclError ret = aclrtCreateEventWithFlag(&event_, ACL_EVENT_SYNC);
    CHECK(ret == ACL_SUCCESS) << "aclrtCreateEventWithFlag failed: " << ret;
    ret = aclrtRecordEvent(event_, stream);
    CHECK(ret == ACL_SUCCESS) << "aclrtRecordEvent failed: " << ret;
  }

  ~NpuForwardOutputReadyEvent() override {
    wait_without_fatal();
    if (event_ != nullptr) {
      const aclError ret = aclrtDestroyEvent(event_);
      if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "aclrtDestroyEvent failed: " << ret;
      }
    }
  }

  void wait() override { wait_impl(/*fatal_on_error=*/true); }

 private:
  void wait_without_fatal() { wait_impl(/*fatal_on_error=*/false); }

  void wait_impl(bool fatal_on_error) {
    std::call_once(wait_once_, [this, fatal_on_error] {
      if (event_ == nullptr) {
        return;
      }
      const aclError ret = aclrtSynchronizeEvent(event_);
      if (ret == ACL_SUCCESS) {
        return;
      }
      if (fatal_on_error) {
        LOG(FATAL) << "aclrtSynchronizeEvent failed: " << ret;
      }
      LOG(ERROR) << "aclrtSynchronizeEvent failed: " << ret;
    });
  }

  aclrtEvent event_ = nullptr;
  std::once_flag wait_once_;
};

}  // namespace

std::shared_ptr<ForwardOutputReadyEvent> record_forward_output_ready_event(
    Stream& stream) {
  return std::make_shared<NpuForwardOutputReadyEvent>(
      stream.get_stream()->stream());
}

#else

std::shared_ptr<ForwardOutputReadyEvent> record_forward_output_ready_event(
    Stream& stream) {
  (void)stream;
  return nullptr;
}

#endif

}  // namespace xllm
