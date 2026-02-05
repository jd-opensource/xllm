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
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/macros.h"
#include "vmm_common.h"

namespace xllm {
namespace vmm {

class VMMWorker;

// VMMSubmitter: Client interface for submitting requests
// Can only be constructed by VMMManager
class VMMSubmitter {
 public:
  ~VMMSubmitter();

  DISALLOW_COPY_AND_MOVE(VMMSubmitter);

  uint64_t map(VirPtr va, PhyMemHandle phy);

  uint64_t unmap(VirPtr va, size_t aligned_size);

  /// Polls completed map/unmap operations from the completion queue.
  /// Called by the submitter thread. Returns the number of completions
  /// processed.
  size_t poll_completions(size_t max_completions = 32);

  bool all_map_done() const;

  bool all_unmap_done() const;

  void wait_all();

  bool is_connected() const { return connected_ && worker_ != nullptr; }

  /// Pushes a completion into the submitter's completion queue.
  /// Called by the worker thread after a map/unmap operation finishes.
  bool push_completion(const VMMCompletion& completion);

 private:
  VMMSubmitter(int32_t device_id);

  bool connect(int32_t device_id);

  void disconnect();

  int32_t device_id_;

  std::shared_ptr<VMMWorker> worker_ = nullptr;

  bool connected_ = false;

  CompletionQueue completion_queue_;

  uint64_t next_request_id_ = 1;

  uint64_t pending_map_ = 0;
  uint64_t pending_unmap_ = 0;

  friend class VMMManager;
};

}  // namespace vmm
}  // namespace xllm