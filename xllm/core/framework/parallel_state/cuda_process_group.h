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

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "process_group.h"

namespace xllm {

class ProcessGroupNccl : public ProcessGroup {
 public:
  ProcessGroupNccl(int global_rank,
                   int world_size,
                   int rank_size,
                   int port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(device) {
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
        c10d::ProcessGroupNCCL::Options::create();
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 7
    pg_options->group_name = group_name;
#endif
    int rank = global_rank;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
      pg_options->global_ranks_in_group = group_ranks;
      rank = local_rank;
    }

    auto store = create_tcp_store(host, port, rank);
    pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
        store, rank, rank_size, pg_options);
  }
};

std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int rank,
    int world_size,
    int rank_size,
    int port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupNccl>(
      rank, world_size, rank_size, port, trans, host, group_name, device);
}
}  // namespace xllm
