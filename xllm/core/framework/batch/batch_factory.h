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

  static BatchFactory* get_instance(int32_t dp_size) {
    CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";
    static std::mutex mu;
    static std::unordered_map<int32_t, std::unique_ptr<BatchFactory, Deleter>>
        instances;
    std::lock_guard<std::mutex> lock(mu);
    auto it = instances.find(dp_size);
    if (it == instances.end()) {
      it = instances
               .emplace(dp_size,
                        std::unique_ptr<BatchFactory, Deleter>(
                            new BatchFactory(dp_size)))
               .first;
    }
    VLOG(1) << "[DIRTY_TRACE][BatchFactory::get_instance] "
            << "requested_dp_size=" << dp_size
            << ", instance_dp_size=" << it->second->dp_size_
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
  BatchFactory(int32_t dp_size) : dp_size_(dp_size) {}
  ~BatchFactory() = default;

  DISALLOW_COPY_AND_ASSIGN(BatchFactory);

 private:
  int32_t dp_size_;
};
}  // namespace xllm
