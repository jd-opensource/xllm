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

#include <memory>
#include <string>

#include "parallel_args.h"
#include "process_group.h"

namespace xllm {

class CollectiveCommunicator {
 public:
  CollectiveCommunicator(int global_rank,
                         int world_size,
                         int dp_size,
                         int ep_size);
  ~CollectiveCommunicator() = default;

  void create_process_groups(const std::string& master_addr,
                             const torch::Device& device);

  // init communicator and return parallel args.
  const ParallelArgs* parallel_args();

 private:
  std::unique_ptr<ParallelArgs> parallel_args_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ProcessGroup> dp_local_process_group_;
  std::unique_ptr<ProcessGroup> tp_group_;
  std::unique_ptr<ProcessGroup> moe_tp_group_;
  std::unique_ptr<ProcessGroup> moe_ep_group_;
};

}  // namespace xllm