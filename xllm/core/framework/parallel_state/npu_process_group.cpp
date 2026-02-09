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

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

namespace {
inline bool is_npu(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}

torch::Tensor flatten_for_scatter_gather(std::vector<torch::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return torch::empty(sizes, t.options());
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case torch::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case torch::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case torch::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case torch::kLong:
      return HCCL_DATA_TYPE_INT64;
    case torch::kInt:
      return HCCL_DATA_TYPE_INT32;
    case torch::kChar:
      return HCCL_DATA_TYPE_INT8;
    case torch::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      LOG(FATAL) << "Unconvertible HCCL type " << type;
  }
}

void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace xllm {

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
  hccl_pg_options->group_id = group_name;

  int32_t rank = global_rank;
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
ProcessGroupImpl::~ProcessGroupImpl() {
  if (pg_) {
    pg_->shutdown();
  } else {
    HCCLCHECK(HcclCommDestroy(comm_));
  }
  c10_npu::NPUCachingAllocator::emptyCache();
}

ProcessGroupImpl::ProcessGroupImpl(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}

void ProcessGroupImpl::allgather(const torch::Tensor& input,
                                 std::vector<torch::Tensor>& outputs) {
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  CHECK_EQ(outputs.size(), world_size())
      << "outputs should have the same size as world_size";
  check_input(input);
  torch::DeviceGuard device_guard(device());

  torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);

  const auto count = input.numel();
  const auto data_type = to_hccl_data_type(input);

  auto compute_stream = c10_npu::getCurrentNPUStream();

  auto ready = std::make_shared<c10_npu::NPUEvent>();
  ready->record(compute_stream);
  ready->block(comm_stream_);

  c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                             comm_stream_);
  c10_npu::NPUCachingAllocator::recordStream(
      flattened_output.storage().data_ptr(), comm_stream_);

  HCCLCHECK(HcclAllGather(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/flattened_output.data_ptr(),
      /*sendcount=*/count,
      /*datatype=*/data_type,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  auto done = std::make_shared<c10_npu::NPUEvent>();
  done->record(comm_stream_);
  done->block(compute_stream);

  for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
    outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
  }
}

void ProcessGroupImpl::allreduce(torch::Tensor& input) {
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  check_input(input);
  torch::DeviceGuard device_guard(device());

  const auto count = input.numel();
  const auto data_type = to_hccl_data_type(input);

  auto compute_stream = c10_npu::getCurrentNPUStream();

  auto ready = std::make_shared<c10_npu::NPUEvent>();
  ready->record(compute_stream);
  ready->block(comm_stream_);

  c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                             comm_stream_);

  HCCLCHECK(HcclAllReduce(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/input.data_ptr(),
      /*count=*/count,
      /*datatype=*/data_type,
      /*op=*/HCCL_REDUCE_SUM,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  auto done = std::make_shared<c10_npu::NPUEvent>();
  done->record(comm_stream_);
  done->block(compute_stream);
}

}  // namespace xllm
