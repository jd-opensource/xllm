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

#include "collective_communicator.h"

#include "mapping_npu.h"

#if defined(USE_NPU)
#include "npu_process_group.h"
#include "xllm_kernels/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_kernels/core/include/atb_speed/utils/singleton.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA)
#include "cuda_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#endif
#include "common/global_flags.h"
#include "parallel_args.h"
#include "process_group.h"
#include "util/net.h"

namespace xllm {

CollectiveCommunicator::CollectiveCommunicator(int global_rank,
                                               int world_size,
                                               int dp_size,
                                               int ep_size) {
#if defined(USE_NPU)
  // create hccl process group with hccl_root_info
  // std::vector<HcclRootInfo> unique_ids;
  // for (const auto& protoId : uids.comm_unique_ids()) {
  //   HcclRootInfo id;
  //   std::memcpy(
  //       id.internal, protoId.comm_unique_id().data(), sizeof(id.internal));
  //   unique_ids.push_back(id);
  // }
  // HcclComm comm;
  // auto hccl_result = HcclCommInitRootInfo(
  //     world_size, &unique_ids[0], global_rank, &comm);
  // CHECK(hccl_result == HCCL_SUCCESS)
  //     << "HcclCommInitRootInfo failed, global rank is " <<
  //     global_rank;
  // std::unique_ptr<ProcessGroupHCCL> hccl_pg =
  //     std::make_unique<ProcessGroupHCCL>(
  //         global_rank, world_size, device, comm);

  // comunicator will be inited in torch.
  if (FLAGS_npu_kernel_backend == "TORCH") {
    parallel_args_ = std::make_unique<ParallelArgs>(
        global_rank, world_size, dp_size, nullptr, ep_size);
    return;
  }

  // comunicator will be inited in atb.
  MappingNPU::Options mapping_options;
  mapping_options.dp_size(dp_size)
      .tp_size(world_size / dp_size)
      .moe_tp_size(world_size / ep_size)
      .moe_ep_size(ep_size)
      .pp_size(1)
      .sp_size(1);
  MappingNPU mapping_npu(
      FLAGS_rank_tablefile, world_size, global_rank, mapping_options);
  auto mapping_data = mapping_npu.to_json();
  atb_speed::base::Mapping mapping;
  mapping.ParseParam(mapping_data);
  mapping.InitGlobalCommDomain(FLAGS_communication_backend);
  auto moeEpParallelInfo = mapping.Get(atb_speed::base::MOE_EP);
  auto dispatchAndCombinecommDomain =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
          moeEpParallelInfo.groupId,
          moeEpParallelInfo.rankIds,
          moeEpParallelInfo.rank,
          FLAGS_communication_backend,
          moeEpParallelInfo.bufferSize,
          false);
  auto dispatchAndCombineHcclComm =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(
          dispatchAndCombinecommDomain);
  parallel_args_ = std::make_unique<ParallelArgs>(global_rank,
                                                  world_size,
                                                  dp_size,
                                                  nullptr,
                                                  ep_size,
                                                  mapping_data,
                                                  mapping,
                                                  dispatchAndCombinecommDomain,
                                                  dispatchAndCombineHcclComm);
#else
  parallel_args_ = std::make_unique<ParallelArgs>(
      global_rank, world_size, dp_size, nullptr, ep_size);
#endif
}

void CollectiveCommunicator::create_process_groups(
    const std::string& master_addr,
    const torch::Device& device) {
#if defined(USE_NPU)
  if (FLAGS_npu_kernel_backend == "ATB") {
    return;
  }
#endif
  std::string host;
  int port;
  net::parse_host_port_from_addr(master_addr, host, port);

  int global_rank = parallel_args_->rank();
  int world_size = parallel_args_->world_size();
  int dp_size = parallel_args_->dp_size();
  int ep_size = parallel_args_->ep_size();
  process_group_ = create_process_group(global_rank,
                                        world_size,
                                        world_size,
                                        ++port,
                                        false,
                                        host,
                                        "world_group",
                                        device);
  parallel_args_->process_group_ = process_group_.get();

  int tp_size = world_size / dp_size;
  CHECK_EQ(tp_size * dp_size, world_size);
  int port_offset = global_rank / tp_size + 1;
  tp_group_ = create_process_group(global_rank,
                                   world_size,
                                   tp_size,
                                   port + port_offset,
                                   false,
                                   host,
                                   "tp_group",
                                   device);
  parallel_args_->tp_group_ = tp_group_.get();
  port += dp_size;

  if (dp_size > 1) {
    port_offset = global_rank % tp_size + 1;
    dp_local_process_group_ = create_process_group(global_rank,
                                                   world_size,
                                                   dp_size,
                                                   port + port_offset,
                                                   true,
                                                   host,
                                                   "dp_group",
                                                   device);
    parallel_args_->dp_local_process_group_ = dp_local_process_group_.get();
    port += tp_size;
  }

  if (ep_size > 1) {
    int moe_tp_size = world_size / ep_size;
    port_offset = global_rank / moe_tp_size + 1;
    moe_tp_group_ = create_process_group(global_rank,
                                         world_size,
                                         moe_tp_size,
                                         port + port_offset,
                                         false,
                                         host,
                                         "moe_tp_group",
                                         device);
    parallel_args_->moe_tp_group_ = moe_tp_group_.get();
    port += ep_size;
    port_offset = global_rank % moe_tp_size + 1;
    moe_ep_group_ = create_process_group(global_rank,
                                         world_size,
                                         ep_size,
                                         port + port_offset,
                                         true,
                                         host,
                                         "moe_ep_group",
                                         device);
    parallel_args_->moe_ep_group_ = moe_ep_group_.get();
  }
}

const ParallelArgs* CollectiveCommunicator::parallel_args() {
  // TODO: init communicator
  return parallel_args_.get();
}

}  // namespace xllm
