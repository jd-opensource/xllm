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

#include <folly/concurrency/UnboundedQueue.h>
#include <stdint.h>

#include "vmm/vmm_api.h"

namespace xllm {
namespace vmm {

enum class OpType { MAP, UNMAP };

class VMMSubmitter;

struct VMMRequest {
  OpType op_type;
  VirPtr va;
  PhyMemHandle phy;
  size_t size = 0;
  uint64_t request_id = 0;
  VMMSubmitter* submitter = nullptr;

  VMMRequest() = default;

  VMMRequest(OpType type,
             VirPtr v,
             PhyMemHandle p,
             size_t s,
             uint64_t id,
             VMMSubmitter* sub)
      : op_type(type), va(v), phy(p), size(s), request_id(id), submitter(sub) {}
};

struct VMMCompletion {
  uint64_t request_id;
  OpType op_type;
  bool success;

  VMMCompletion() : request_id(0), op_type(OpType::MAP), success(false) {}

  VMMCompletion(uint64_t id, OpType type, bool succ)
      : request_id(id), op_type(type), success(succ) {}
};

using RequestQueue = folly::UMPSCQueue<VMMRequest, /* Mayblock */ false>;
using CompletionQueue = folly::USPSCQueue<VMMCompletion, /* Mayblock */ false>;

}  // namespace vmm
}  // namespace xllm