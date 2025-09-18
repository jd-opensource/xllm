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
#include <absl/container/flat_hash_set.h>

#include <memory>

#include "call.h"
#include "core/runtime/llm_master.h"

namespace xllm {

template <typename T>
class APIServiceImpl {
 public:
  APIServiceImpl(LLMMaster* master, const std::vector<std::string>& models)
      : master_(master), models_(models.begin(), models.end()) {
    CHECK(master != nullptr);
    CHECK(!models_.empty());
  }
  virtual ~APIServiceImpl() = default;

  void process_async(std::shared_ptr<Call> call) {
    std::shared_ptr<T> call_cast = std::dynamic_pointer_cast<T>(call);
    process_async_impl(call_cast);
  }

  virtual void process_async_impl(std::shared_ptr<T> call) = 0;

 protected:
  LLMMaster* master_;
  absl::flat_hash_set<std::string> models_;
};

}  // namespace xllm
