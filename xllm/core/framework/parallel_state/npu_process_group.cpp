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

#include "npu_process_group.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

namespace {

#define HCCLCHECK(cmd)                                               \
  do {                                                               \
    HcclResult r = cmd;                                              \
    if (r != HCCL_SUCCESS) {                                         \
      LOG(FATAL) << "Failed, HCCL error :" << HcclGetErrorString(r); \
    }                                                                \
  } while (0)
}  // namespace

namespace xllm {

ProcessGroupHCCL::ProcessGroupHCCL(int global_rank,
                                   int world_size,
                                   int rank_size,
                                   int port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(device) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
  // hccl_pg_options->group_name = group_name;
  int rank = global_rank;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, global_rank, rank_size, trans);
    std::vector<uint32_t> uint32_ranks;
    for (auto rank : group_ranks) {
      uint32_ranks.push_back(static_cast<uint32_t>(rank));
    }
    hccl_pg_options->global_ranks_in_group = uint32_ranks;
    rank = local_rank;
  }

  auto store = create_tcp_store(host, port, rank);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, rank, rank_size, hccl_pg_options);
}

// Destructor.
ProcessGroupHCCL::~ProcessGroupHCCL() {
  if (pg_) {
    pg_->shutdown();
  } else {
    HCCLCHECK(HcclCommDestroy(comm_));
  }
}

ProcessGroupHCCL::ProcessGroupHCCL(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(device), comm_(comm) {}

std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int rank,
    int world_size,
    int rank_size,
    int port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupHCCL>(
      rank, world_size, rank_size, port, trans, host, group_name, device);
}

}  // namespace xllm