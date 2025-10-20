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

#include "mlu_process_group.h"

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace {

std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size) {
  int target_group_index = global_rank / split_size;
  uint64_t start_rank = target_group_index * split_size;
  uint64_t end_rank = start_rank + split_size;
  std::vector<uint64_t> group_rank;
  int index = global_rank - start_rank;
  for (uint64_t rank = start_rank; rank < end_rank; rank++) {
    group_rank.push_back(rank);
  }
  return {index, group_rank};
}

}  // namespace

namespace xllm {

ProcessGroupCncl::ProcessGroupCncl(int rank,
                                   int world_size,
                                   int rank_size,
                                   int port,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(rank, rank_size, device),
      world_size_(rank_size),
      rank_(rank) {
  c10::intrusive_ptr<torch_mlu::ProcessGroupCNCL::Options> cncl_pg_options =
      torch_mlu::ProcessGroupCNCL::Options::create();
  cncl_pg_options->group_name = group_name;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, rank, rank_size);
    cncl_pg_options->global_ranks_in_group = group_ranks;
    rank_ = local_rank;
  }

  c10d::TCPStoreOptions tcp_options;
  tcp_options.isServer = (rank_ == 0);
  tcp_options.port = port;

  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<c10d::TCPStore>(host, tcp_options);
  cncl_pg_ = std::make_unique<torch_mlu::ProcessGroupCNCL>(
      store, rank, world_size, cncl_pg_options);
}

// Destructor.
ProcessGroupCncl::~ProcessGroupCncl() { cncl_pg_->shutdown(); }

void ProcessGroupCncl::allreduce(torch::Tensor& input) {
  std::vector<torch::Tensor> input_tensors = {input};
  cncl_pg_->allreduce(input_tensors)->wait();
}

void ProcessGroupCncl::allgather(torch::Tensor input,
                                 std::vector<torch::Tensor>& outputs) {
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<std::vector<torch::Tensor>> output_tensors = {outputs};
  cncl_pg_->allgather(output_tensors, input_tensors)->wait();
}
}  // namespace xllm