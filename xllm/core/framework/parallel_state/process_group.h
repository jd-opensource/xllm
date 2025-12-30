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

#if defined(USE_NPU)
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>
#endif

namespace xllm {

class ProcessGroupImpl;

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

  // reduce_scatter: reduce the input tensor across all processes, scatter the
  // reduced chunks to all processes so that each process gets one chunk of the
  // result. we use default dim 0 for reduce_scatter.
  virtual void reduce_scatter(const torch::Tensor& input,
                              torch::Tensor& output);

 private:
  // device of current process
  torch::Device device_;

 protected:
#if defined(USE_NPU) &&         \
    (TORCH_VERSION_MAJOR < 2 || \
     (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 7))
  // Using ProcessGroupHCCL for NPU devices
  // Note: torch_npu uses an older torch version where c10d::Backend lacks
  // shutdown() method
  std::unique_ptr<c10d_npu::ProcessGroupHCCL> pg_{nullptr};
#else
  std::unique_ptr<c10d::Backend> pg_{nullptr};
#endif
};

std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device);

}  // namespace xllm