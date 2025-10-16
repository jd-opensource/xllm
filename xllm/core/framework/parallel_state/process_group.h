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

namespace xllm {

class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, const torch::Device& device)
      : rank_(rank), world_size_(world_size), device_(device) {}

  virtual ~ProcessGroup() = default;

  virtual int rank() { return rank_; }

  virtual int world_size() { return world_size_; }

  const torch::Device& device() const { return device_; }

  // allreduce: reduce the input tensor across all processes, and all processes
  // get the result.
  virtual void allreduce(torch::Tensor& input) = 0;

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(torch::Tensor input,
                         std::vector<torch::Tensor>& outputs) = 0;

 private:
  // rank of current process.
  int rank_ = 0;

  // number of processes.
  int world_size_ = 0;

  // device of current process.
  torch::Device device_;
};

}  // namespace xllm