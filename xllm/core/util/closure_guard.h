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

#include <google/protobuf/service.h>

#include <functional>

#include "butil/macros.h"

namespace xllm {

// RAII: Call Run() of the closure on destruction.
class ClosureGuard {
 public:
  ClosureGuard() : _done(NULL) {}

  // Constructed with a closure which will be Run() inside dtor.
  explicit ClosureGuard(google::protobuf::Closure* done,
                        std::function<void(void*)>&& before_done,
                        std::function<void(void*)>&& after_done)
      : _done(done), _before_done(before_done), _after_done(after_done) {
    _before_done(nullptr);
  }

  // Run internal closure if it's not NULL.
  ~ClosureGuard() {
    if (_done) {
      _after_done(nullptr);
      _done->Run();
    }
  }

  // Run internal closure if it's not NULL and set it to `done'.
  void reset(google::protobuf::Closure* done) {
    if (_done) {
      _done->Run();
    }
    _done = done;
  }

  // Return and set internal closure to NULL.
  google::protobuf::Closure* release() {
    _after_done(nullptr);

    google::protobuf::Closure* const prev_done = _done;
    _done = NULL;
    return prev_done;
  }

  // True if no closure inside.
  bool empty() const { return _done == NULL; }

 private:
  // Copying this object makes no sense.
  DISALLOW_COPY_AND_ASSIGN(ClosureGuard);

  google::protobuf::Closure* _done;
  std::function<void(void*)> _before_done;
  std::function<void(void*)> _after_done;
};

}  // namespace xllm
