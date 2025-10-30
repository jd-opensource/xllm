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

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
namespace xllm {
std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size,
                                                     bool trans);

c10::intrusive_ptr<c10d::Store> create_tcp_store(const std::string& host,
                                                 int port,
                                                 int rank);

class ProcessGroup {
 public:
  ProcessGroup(const torch::Device& device) : device_(device) {}

  virtual ~ProcessGroup() = default;

  int rank() const {
    CHECK(pg_ != nullptr) << "Process group is not initialized.";
    return pg_->getRank();
  }

  int world_size() const {
    CHECK(pg_ != nullptr) << "Process group is not initialized.";
    return pg_->getSize();
  }

  const torch::Device& device() const { return device_; }

  // allreduce: reduce the input tensor across all processes, and all processes
  // get the result.
  virtual void allreduce(torch::Tensor& input);

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(const torch::Tensor& input,
                         std::vector<torch::Tensor>& outputs);

 private:
  // device of current process
  torch::Device device_;

 protected:
  std::unique_ptr<c10d::Backend> pg_{nullptr};
};

}  // namespace xllm