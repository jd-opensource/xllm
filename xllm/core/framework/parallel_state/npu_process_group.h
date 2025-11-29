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

#include "hccl/hccl.h"
#include "process_group.h"

namespace xllm {

class ProcessGroupHCCL : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupHCCL(int rank,
                   int world_size,
                   const torch::Device& device,
                   HcclComm comm);

  ProcessGroupHCCL(int rank,
                   int world_size,
                   int rank_size,
                   int port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  // Destructor.
  ~ProcessGroupHCCL() override;

 private:
  HcclComm comm_ = nullptr;
};

std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int rank,
    int world_size,
    int rank_size,
    int port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device);

}  // namespace xllm