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

#include <brpc/controller.h>

#include <string>

namespace xllm {

class Call {
 public:
  Call(brpc::Controller* controller);
  virtual ~Call() = default;

  std::string get_x_request_id() { return x_request_id_; }
  std::string get_x_request_time() { return x_request_time_; }

  bool get_binary_payload(std::string& payload);

  virtual bool is_disconnected() const = 0;

 protected:
  void init();

 protected:
  brpc::Controller* controller_;

  std::string x_request_id_;
  std::string x_request_time_;
};

}  // namespace xllm
