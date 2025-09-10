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
#include <functional>
#include <memory>
#include <string>

#include "common.pb.h"
#include "framework/request/request.h"

namespace xllm {
class PriorityComparator {
 public:
  virtual bool operator()(const std::shared_ptr<Request>& a,
                          const std::shared_ptr<Request>& b) const = 0;
  virtual ~PriorityComparator() = default;
};

struct FCFSComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct StrictPriorityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DeadlineComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy);

}  // namespace xllm