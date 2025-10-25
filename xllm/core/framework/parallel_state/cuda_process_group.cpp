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

#include "cuda_process_group.h"

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "parallel_state.h"

namespace xllm {

ProcessGroupNccl::ProcessGroupNccl(int rank,
                                   int world_size,
                                   int rank_size,
                                   int port,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(rank, rank_size, device),
      world_size_(rank_size),
      rank_(rank) {
  c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> nccl_pg_options =
      c10d::ProcessGroupNCCL::Options::create();
  nccl_pg_options->is_high_priority_stream = false;

  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        parallel_state::get_group_rank(world_size, rank, rank_size);
    nccl_pg_options->global_ranks_in_group = group_ranks;
    rank_ = local_rank;
  }

  c10d::TCPStoreOptions tcp_options;
  tcp_options.isServer = (rank_ == 0);
  tcp_options.port = port;

  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<c10d::TCPStore>(host, tcp_options);
  nccl_pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
      store, rank_, rank_size, nccl_pg_options);
}

ProcessGroupNccl::~ProcessGroupNccl() { nccl_pg_->shutdown(); }

void ProcessGroupNccl::allreduce(torch::Tensor& input) {
  std::vector<torch::Tensor> input_tensors = {input};
  nccl_pg_->allreduce(input_tensors)->wait();
}

void ProcessGroupNccl::allgather(torch::Tensor input,
                                 std::vector<torch::Tensor>& outputs) {
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<std::vector<torch::Tensor>> output_tensors = {outputs};
  nccl_pg_->allgather(output_tensors, input_tensors)->wait();
}

}  // namespace xllm
