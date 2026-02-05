/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <thread>
#include <unordered_set>

#include "core/common/macros.h"
#include "vmm_common.h"

namespace xllm {
namespace vmm {
// VMMWorker: Worker thread that executes VMM operations
// Can only be constructed by VMMManager
class VMMWorker {
 public:
  ~VMMWorker();

  DISALLOW_COPY_AND_MOVE(VMMWorker);

  void start();

  void stop();

  bool submit_request(const VMMRequest& req);

 private:
  VMMWorker(int32_t device_id);

  void worker_loop();

  bool step_current();

  bool step_deferred();

  void defer_request(const VMMRequest& req);

  void schedule(int32_t max_ops);

  bool has_conflict(VirPtr va);

  void execute_map(VMMRequest& req);

  void execute_unmap(VMMRequest& req);

  void notify_completion(VMMSubmitter* submitter,
                         uint64_t request_id,
                         OpType op_type,
                         bool success);

  int32_t device_id_;
  std::unique_ptr<std::thread> worker_thread_;
  std::atomic<bool> running_;

  RequestQueue work_queue_;

  std::unordered_set<VirPtr> deferred_va_;

  std::deque<VMMRequest> deferred_requests_;

  friend class VMMManager;
};

}  // namespace vmm
}  // namespace xllm