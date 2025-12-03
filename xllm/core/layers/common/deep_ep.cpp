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

#include "deep_ep.h"

#include <glog/logging.h>

#include "framework/parallel_state/process_group.h"
#include "kernels/ops_api.h"

namespace {
/**
 * @brief AllGather a metadata tensor on CPU by using Device communication
 *
 * Transfers local tensor from CPU to Device, performs Device AllGather,
 * and copies gathered results back to the allocated CPU output tensor.
 *
 * @param process_group ProcessGroup pointer
 * @param local_exchange_info_cpu Local metadata tensor on CPU [M]
 * @param all_exchange_info_cpu Output metadata tensor on CPU [nranks, M],
 * pre-allocated
 * @param device Device for communication
 */
void offload_exchange_info_allgather(
    xllm::ProcessGroup* process_group,
    const torch::Tensor& local_exchange_info_cpu,
    torch::Tensor& all_exchange_info_cpu,
    const torch::Device& device) {
  // Checks for shape and device
  CHECK(local_exchange_info_cpu.device().is_cpu())
      << "Input tensor must be on CPU";
  CHECK(all_exchange_info_cpu.device().is_cpu())
      << "Output tensor must be on CPU";

  int64_t nranks = all_exchange_info_cpu.size(0);
  int64_t info_size = local_exchange_info_cpu.size(0);

  // Copy input from CPU to device
  torch::Tensor local_tensor_device = local_exchange_info_cpu.to(device);

  // Prepare allgather input/output containers
  std::vector<torch::Tensor> device_outputs;
  device_outputs.reserve(nranks);

  // Allocate device tensors for all_exchange_info_cpu of each rank
  for (int64_t i = 0; i < nranks; ++i) {
    device_outputs.push_back(torch::empty_like(local_tensor_device));
  }

  // Perform device AllGather and wait for completion
  process_group->allgather(local_tensor_device, device_outputs);

  // Copy gathered device tensors back to all_exchange_info_cpu
  for (int64_t i = 0; i < nranks; ++i) {
    torch::Tensor row_cpu = all_exchange_info_cpu.select(0, i);
    row_cpu.copy_(device_outputs[i]);
  }
}
}  // namespace

namespace xllm {
namespace layer {

DeepEPImpl::DeepEPImpl(int64_t dispatch_token_size,
                       int64_t combine_token_size,
                       int64_t max_num_tokens_per_rank,
                       int64_t num_global_experts,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options)
    : max_num_tokens_per_rank_(max_num_tokens_per_rank),
      parallel_args_(parallel_args),
      options_(options) {
  ProcessGroup* moe_ep_group = parallel_args.moe_ep_group_;
  int32_t rank = parallel_args.rank_;
  int32_t nranks = parallel_args.world_size_;

  // Step 1: Call moe_all2all_create to obtain the deep ep handle and buffer
  // tensors
  //  for All-to-All communication.
  // Only needs to be done once.
  xllm::kernel::MoeAll2AllCreateParams create_params;
  create_params.dispatch_token_byte = dispatch_token_size;
  create_params.combine_token_byte = combine_token_size;
  create_params.max_expert_num = num_global_experts;
  create_params.max_token_num = max_num_tokens_per_rank;
  create_params.rank = rank;
  create_params.nrank = nranks;
  create_params.device = options_.device();

  // moe_all2all_create returns a vector of tensors:
  // index 0 is a tensor contains [handle, exchange_info_size]
  // Index 1: exchange_info
  // Index 2: dispatch_send
  // Index 3: dispatch_recv
  // Index 4: combine_send
  // Index 5: combine_recv
  std::vector<torch::Tensor> created_tensors =
      xllm::kernel::moe_all2all_create(create_params);
  CHECK(created_tensors.size() == 6)
      << "moe_all2all_create returned incorrect number of tensors";
  // Extract handle and exchange_info_size from created_tensors[0]
  handle_ = created_tensors[0][0].item<int64_t>();
  torch::Tensor exchange_info = created_tensors[1];

  // register buffers to the module
  dispatch_send_token_tensor_ =
      register_buffer("dispatch_send_token", created_tensors[2]);
  dispatch_recv_token_tensor_ =
      register_buffer("dispatch_recv_token", created_tensors[3]);
  combine_send_token_tensor_ =
      register_buffer("combine_send_token", created_tensors[4]);
  combine_recv_token_tensor_ =
      register_buffer("combine_recv_token", created_tensors[5]);

  // Step 2: Gather all_exchange_info by performing an All-Gather operation
  //  on exchange_info across nrank processes.
  // Only needs to be done once.
  auto cpu_options =
      torch::TensorOptions().dtype(exchange_info.dtype()).device(torch::kCPU);
  torch::Tensor all_exchange_info =
      torch::empty({nranks, exchange_info.size(0)}, cpu_options);
  offload_exchange_info_allgather(
      moe_ep_group, exchange_info, all_exchange_info, options_.device());

  // Step 3: moe_all2all_init to configure the all_exchange_info into the
  // handle. Only needs to be done once.
  xllm::kernel::MoeAll2AllInitParams init_params;
  init_params.handle = handle_;
  init_params.all_exchange_info = all_exchange_info;
  init_params.device = options_.device();
  xllm::kernel::moe_all2all_init(init_params);

  is_initialized_ = true;
}

DeepEPImpl::~DeepEPImpl() {
  if (is_initialized_) {
    xllm::kernel::MoeAll2AllDestroyParams destroy_params;
    destroy_params.handle = handle_;
    destroy_params.device = options_.device();
    xllm::kernel::moe_all2all_destroy(destroy_params);
  }
}

void DeepEPImpl::dispatch(int64_t token_byte,
                          int64_t token_num,
                          const torch::Tensor& send_layout,
                          const torch::Tensor& send_token_num,
                          const torch::Tensor& recv_layout,
                          const torch::Tensor& recv_token_num,
                          const std::optional<torch::Tensor>& send_token,
                          const std::optional<torch::Tensor>& recv_token) {
  xllm::kernel::MoeAll2AllDispatchParams params;
  params.handle = handle_;
  params.token_byte = token_byte;
  params.token_num = token_num;
  params.send_layout = send_layout;
  params.send_token_num = send_token_num;
  params.recv_layout = recv_layout;
  params.recv_token_num = recv_token_num;
  params.send_token = send_token;
  params.recv_token = recv_token;

  xllm::kernel::moe_all2all_dispatch(params);
}

void DeepEPImpl::combine(int64_t token_byte,
                         int64_t token_num,
                         const torch::Tensor& send_src_layout,
                         const torch::Tensor& send_dst_layout,
                         const std::optional<torch::Tensor>& send_token,
                         const std::optional<torch::Tensor>& recv_token) {
  xllm::kernel::MoeAll2AllCombineParams params;
  params.handle = handle_;
  params.token_byte = token_byte;
  params.token_num = token_num;
  params.send_src_layout = send_src_layout;
  params.send_dst_layout = send_dst_layout;
  params.send_token = send_token;
  params.recv_token = recv_token;

  xllm::kernel::moe_all2all_combine(params);
}

DeepEPBuffer DeepEPImpl::get_buffer() const {
  return DeepEPBuffer{dispatch_send_token_tensor_,
                      dispatch_recv_token_tensor_,
                      combine_send_token_tensor_,
                      combine_recv_token_tensor_};
}

}  // namespace layer
}  // namespace xllm
