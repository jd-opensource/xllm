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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/metrics.h"
#include "framework/batch/batch.h"

namespace xllm {

class BatchFactory {
 public:
  struct Deleter {
    void operator()(BatchFactory* ptr) const { delete ptr; }
  };

  static BatchFactory* get_instance(int32_t dp_size, int32_t cp_size = 1) {
    CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";
    CHECK_GT(cp_size, 0) << "cp_size must be greater than 0";
    static std::mutex mu;
    static std::unordered_map<int64_t, std::unique_ptr<BatchFactory, Deleter>>
        instances;
    std::lock_guard<std::mutex> lock(mu);
    const int64_t key = (static_cast<int64_t>(dp_size) << 32) |
                        static_cast<uint32_t>(cp_size);
    auto it = instances.find(key);
    if (it == instances.end()) {
      it = instances
               .emplace(key,
                        std::unique_ptr<BatchFactory, Deleter>(
                            new BatchFactory(dp_size, cp_size)))
               .first;
    }
    VLOG(1) << "[DIRTY_TRACE][BatchFactory::get_instance] "
            << "requested_dp_size=" << dp_size
            << ", requested_cp_size=" << cp_size
            << ", instance_dp_size=" << it->second->dp_size_
            << ", instance_cp_size=" << it->second->cp_size_
            << ", instance_ptr=" << static_cast<const void*>(it->second.get());
    return it->second.get();
  }

  std::vector<Batch> create_batches(
      const std::vector<std::shared_ptr<Request>>& running_requests,
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      // for beam-search
      std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos =
          nullptr);

  std::vector<Batch> create_rec_batches(
      const std::vector<std::shared_ptr<Request>>& running_requests,
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos =
          nullptr);

 private:
  BatchFactory(int32_t dp_size, int32_t cp_size)
      : dp_size_(dp_size), cp_size_(cp_size) {}
  ~BatchFactory() = default;

  DISALLOW_COPY_AND_ASSIGN(BatchFactory);

 private:
  int32_t dp_size_;
  int32_t cp_size_;
};
}  // namespace xllm
