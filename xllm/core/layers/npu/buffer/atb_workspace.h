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

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "atb_buffer.h"

namespace xllm {

class AtbWorkspace {
 public:
  AtbWorkspace() = default;

  AtbWorkspace(at::Device device);

  ~AtbWorkspace();

  AtbWorkspace(const AtbWorkspace&) = delete;

  AtbWorkspace& operator=(const AtbWorkspace&) = delete;

  AtbWorkspace(AtbWorkspace&&) = default;

  AtbWorkspace& operator=(AtbWorkspace&&) = default;

  void* GetWorkspaceBuffer(uint64_t bufferSize);

 private:
  static std::map<int32_t, std::unique_ptr<AtbBuffer>> buffer_map_;
};

}  // namespace xllm
