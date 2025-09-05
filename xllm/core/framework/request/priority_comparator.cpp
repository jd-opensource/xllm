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

#include "priority_comparator.h"

#include "glog/logging.h"

namespace xllm {

// implement operator()
bool FCFSComparator::operator()(const std::shared_ptr<Request>& a,
                                const std::shared_ptr<Request>& b) const {
  return a->created_time() > b->created_time();
}

bool StrictPriorityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto priority_a = a->priority();
  auto priority_b = b->priority();
  if (priority_a != priority_b) {
    return priority_a > priority_b;  // HIGH(1) < NORMAL(2) < LOW(3)
  }
  return a->created_time() > b->created_time();
}

bool DeadlineComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {
  return a->slo_ms() - a->elapsed_seconds() * 1000 >
         b->slo_ms() - b->elapsed_seconds() * 1000;
}

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy) {
  if (priority_strategy == "FCFS") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return FCFSComparator()(a, b);
    };
  } else if (priority_strategy == "priority") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b);
    };
  } else if (priority_strategy == "deadline") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return DeadlineComparator()(a, b);
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

}  // namespace xllm