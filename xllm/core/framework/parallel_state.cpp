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

#include "parallel_state.h"

#include <c10/core/Device.h>
#if defined(USE_NPU)
#include <hccl/hccl_types.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "hccl/hccl.h"
#endif
#pragma GCC diagnostic pop
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm {

namespace {
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                                               \
  do {                                                               \
    HcclResult r = cmd;                                              \
    if (r != HCCL_SUCCESS) {                                         \
      LOG(FATAL) << "Failed, HCCL error :" << HcclGetErrorString(r); \
    }                                                                \
  } while (0)
#endif
inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}
inline bool is_npu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}
inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}
at::Tensor flatten_for_scatter_gather(std::vector<at::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}
#if defined(USE_NPU)
HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case at::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case at::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case at::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case at::kLong:
      return HCCL_DATA_TYPE_INT64;
    case at::kInt:
      return HCCL_DATA_TYPE_INT32;
    case at::kChar:
      return HCCL_DATA_TYPE_INT8;
    case at::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      TORCH_CHECK(false, "Unconvertible HCCL type ", type);
  }
}
#endif
void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace parallel_state {

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  const auto rank = parallel_args.rank();
  // auto* process_group = parallel_args.process_group();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  // process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/-1).contiguous();
}

torch::Tensor reduce(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }
  // auto* process_group = parallel_args.process_group();
  // process_group->allreduce(input);
  return input;
}

torch::Tensor scatter(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  // get the size for last dimension
  const auto last_dim_size = input.size(-1);
  CHECK(last_dim_size % world_size == 0)
      << "last_dim_size " << last_dim_size << " not divisible by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(last_dim_size / world_size, /*dim=*/-1);
  const auto rank = parallel_args.rank();
  return tensor_list[rank];
}

}  // namespace parallel_state

#if defined(USE_NPU)
std::vector<std::unique_ptr<ProcessGroup>> ProcessGroup::create_process_groups(
    const std::vector<torch::Device>& devices) {
  CHECK(!devices.empty()) << "devices should not be empty";
  for (const auto& device : devices) {
    CHECK(is_npu(device)) << "device should be npu device";
  }
  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }
  std::vector<HcclComm> comms(devices.size());
  const int world_size = static_cast<int>(devices.size());
  // HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(),comms.data()));
  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupHCCL>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
  return process_groups;
}
#elif defined(USE_MLU)
// TODO(mlu): implement create_process_groups for mlu
std::vector<std::unique_ptr<ProcessGroup>> ProcessGroup::create_process_groups(
    const std::vector<torch::Device>& devices) {
  return {};
}
#endif

#if defined(USE_NPU)
ProcessGroupHCCL::ProcessGroupHCCL(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device), comm_(comm) {}
// Destructor.
ProcessGroupHCCL::~ProcessGroupHCCL() { HCCLCHECK(HcclCommDestroy(comm_)); }

void ProcessGroupHCCL::allreduce(torch::Tensor& input) {
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";
  check_input(input);
  // inplace all reduce
  // const auto count = input.numel();
  // const auto data_type = to_hccl_data_type(input);
  // auto stream = c10_npu::getCurrentNPUStream();
  // torch::DeviceGuard device_guard(device());
  // HCCLCHECK(HcclAllReduce(
  //     /*sendbuff=*/input.data_ptr(),
  //     /*recvbuff=*/input.data_ptr(),
  //     /*count=*/count,
  //     /*datatype=*/data_type,
  //     /*op=*/HCCL_REDUCE_SUM,
  //     /*comm=*/comm_,
  //     /*stream=*/stream));
}
void ProcessGroupHCCL::allgather(torch::Tensor input,
                                 std::vector<torch::Tensor>& outputs) {
  check_input(input);
  // CHECK(outputs.size() == world_size())
  //     << "outputs should have the same size as world_size";
  // DCHECK(input.device() == device())
  //     << "input should be on the same device as the process group";
  // torch::DeviceGuard device_guard(device());
  // torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);
  // const auto count = input.numel();
  // const auto data_type = to_hccl_data_type(input);
  // auto stream = c10_npu::getCurrentNPUStream();
  // HCCLCHECK(HcclAllGather(
  //     /*sendbuff=*/input.data_ptr(),
  //     /*recvbuff=*/flattened_output.data_ptr(),
  //     /*sendcount=*/count,
  //     /*datatype=*/data_type,
  //     /*comm=*/comm_,
  //     /*stream=*/stream));
  // // copy the flattened output tensors to the outputs.
  // for (int i = 0; i < outputs.size(); ++i) {
  //   outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
  // }
}
#endif
}  // namespace xllm
