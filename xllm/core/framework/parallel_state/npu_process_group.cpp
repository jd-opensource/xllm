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

namespace {

#define HCCLCHECK(cmd)                                               \
  do {                                                               \
    HcclResult r = cmd;                                              \
    if (r != HCCL_SUCCESS) {                                         \
      LOG(FATAL) << "Failed, HCCL error :" << HcclGetErrorString(r); \
    }                                                                \
  } while (0)

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

void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}

}  // namespace

namespace xllm {

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
}  // namespace xllm