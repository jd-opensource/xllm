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

#include <torch_mlu/csrc/framework/distributed/process_group_cncl.hpp>

#include "process_group.h"

namespace xllm {

class ProcessGroupCncl : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupCncl(int rank,
                   int world_size,
                   int rank_size,
                   int port,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  int rank() override { return rank_; }

  int world_size() override { return world_size_; }

  // Destructor.
  ~ProcessGroupCncl() override;

  void allreduce(torch::Tensor& input) override;

  void allgather(torch::Tensor input,
                 std::vector<torch::Tensor>& outputs) override;

 private:
  std::shared_ptr<torch_mlu::ProcessGroupCNCL> cncl_pg_ = nullptr;
  // rank of current process.
  int rank_ = 0;

  // number of processes.
  int world_size_ = 0;
};

}  // namespace xllm