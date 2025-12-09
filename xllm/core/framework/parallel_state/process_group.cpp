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

#include "process_group.h"

#if defined(USE_NPU)
#include "npu_process_group.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA)
#include "cuda_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#endif

namespace {
std::pair<int, std::vector<uint64_t>> get_trans_group_rank(int world_size,
                                                           int global_rank,
                                                           int split_size) {
  int trans_group_count = split_size;
  int trans_group_size = world_size / split_size;
  int trans_group_index = global_rank % trans_group_size;
  int trans_index = global_rank / trans_group_size;
  std::vector<uint64_t> trans_group_ranks;
  for (int i = 0; i < trans_group_count; i++) {
    uint64_t rank = i * trans_group_size + trans_group_index;
    trans_group_ranks.push_back(rank);
  }

  return {trans_index, trans_group_ranks};
}
}  // namespace

namespace xllm {

std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size,
                                                     bool trans) {
  if (trans) {
    return get_trans_group_rank(world_size, global_rank, split_size);
  }
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

c10::intrusive_ptr<c10d::Store> create_tcp_store(const std::string& host,
                                                 int port,
                                                 int rank) {
  c10d::TCPStoreOptions tcp_options;
  tcp_options.isServer = (rank == 0);
  tcp_options.port = port;
  return c10::make_intrusive<c10d::TCPStore>(host, tcp_options);
}

void ProcessGroup::allreduce(torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  pg_->allreduce(input_tensors)->wait();
}

void ProcessGroup::allgather(const torch::Tensor& input,
                             std::vector<torch::Tensor>& outputs) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<std::vector<torch::Tensor>> output_tensors = {outputs};
  pg_->allgather(output_tensors, input_tensors)->wait();
}

std::unique_ptr<ProcessGroup> create_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupImpl>(
      rank, world_size, rank_size, port, trans, host, group_name, device);
}

}  // namespace xllm
