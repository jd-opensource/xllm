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
#include "xllm_atb_layers/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/singleton.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA)
#include "cuda_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#elif defined(USE_MUSA)
#include "musa_process_group.h"
#endif
#include "common/global_flags.h"
#include "parallel_args.h"
#include "process_group.h"
#include "util/net.h"

namespace xllm {

namespace {

#if defined(USE_NPU)

int get_collective_root_info_count(int world_size, int dp_size, int ep_size) {
  CHECK_GT(world_size, 0);
  CHECK_GT(dp_size, 0);
  CHECK_GT(ep_size, 0);
  CHECK_EQ(world_size % dp_size, 0);
  CHECK_EQ(world_size % ep_size, 0);

  int count = 1 + dp_size;  // world + TP groups
  if (dp_size > 1) {
    count += world_size / dp_size;  // DP-local groups
  }
  if (ep_size > 1) {
    count += ep_size;               // MoE TP groups
    count += world_size / ep_size;  // MoE EP groups
  }
  return count;
}

int get_tp_root_info_index(int global_rank, int world_size, int dp_size) {
  const int tp_size = world_size / dp_size;
  return 1 + global_rank / tp_size;
}

int get_dp_root_info_index(int global_rank, int world_size, int dp_size) {
  const int tp_size = world_size / dp_size;
  return 1 + dp_size + (global_rank % tp_size);
}

int get_moe_tp_root_info_index(int global_rank,
                               int world_size,
                               int dp_size,
                               int ep_size) {
  const int dp_group_count = dp_size > 1 ? (world_size / dp_size) : 0;
  const int moe_tp_size = world_size / ep_size;
  return 1 + dp_size + dp_group_count + (global_rank / moe_tp_size);
}

int get_moe_ep_root_info_index(int global_rank,
                               int world_size,
                               int dp_size,
                               int ep_size) {
  const int dp_group_count = dp_size > 1 ? (world_size / dp_size) : 0;
  const int moe_tp_group_count = ep_size > 1 ? ep_size : 0;
  const int moe_tp_size = world_size / ep_size;
  return 1 + dp_size + dp_group_count + moe_tp_group_count +
         (global_rank % moe_tp_size);
}

std::unique_ptr<ProcessGroup> create_process_group_from_root_info(
    const char* group_name,
    int global_rank,
    int world_size,
    int rank_size,
    bool trans,
    const torch::Device& device,
    const HcclRootInfo& root_info) {
  int local_rank = global_rank;
  if (world_size != rank_size) {
    local_rank =
        get_group_rank(world_size, global_rank, rank_size, trans).first;
  }

  const auto error = aclrtSetDevice(device.index());
  CHECK_EQ(error, ACL_SUCCESS)
      << "ACL set device id " << device.index() << " failed. Error : " << error;

  LOG(INFO) << "Initializing HCCL process group from root info"
            << " [group=" << group_name << ", global_rank=" << global_rank
            << ", local_rank=" << local_rank << ", world_size=" << world_size
            << ", rank_size=" << rank_size << ", trans=" << trans
            << ", device=" << device.str() << "]";

  HcclComm comm = nullptr;
  const auto status =
      HcclCommInitRootInfo(rank_size, &root_info, local_rank, &comm);
  CHECK_EQ(status, HCCL_SUCCESS)
      << "HcclCommInitRootInfo failed for global_rank=" << global_rank
      << ", local_rank=" << local_rank << ", rank_size=" << rank_size
      << ", trans=" << trans;

  LOG(INFO) << "Initialized HCCL process group from root info"
            << " [group=" << group_name << ", global_rank=" << global_rank
            << ", local_rank=" << local_rank << ", rank_size=" << rank_size
            << ", trans=" << trans << ", device=" << device.str()
            << ", comm=" << static_cast<const void*>(comm) << "]";

  return std::make_unique<ProcessGroupImpl>(
      local_rank, rank_size, device, comm);
}

#endif

}  // namespace

#if defined(USE_NPU)
CollectiveCommunicator::CollectiveCommunicator(
    int global_rank,
    int world_size,
    int dp_size,
    int ep_size,
    std::vector<HcclRootInfo> root_infos)
    : root_infos_(std::move(root_infos)) {
#else
CollectiveCommunicator::CollectiveCommunicator(int global_rank,
                                               int world_size,
                                               int dp_size,
                                               int ep_size) {
#endif
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
  // HACK: MappingNPU internally uses a static counter to auto-assign
  // buffer_offset for multi-model scenarios. This is a hack and should be
  // refactored later.
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
  HcclComm dispatchAndCombineHcclComm = nullptr;
  std::string dispatchAndCombinecommDomain;
  moeEpParallelInfo.InitCommDomain(dispatchAndCombineHcclComm,
                                   dispatchAndCombinecommDomain,
                                   FLAGS_communication_backend);
  if (dispatchAndCombineHcclComm == nullptr) {
    LOG(WARNING)
        << "CollectiveCommunicator failed to initialize dispatch/combine comm "
           "through ParallelInfo::InitCommDomain; falling back to direct "
           "ExternalCommManager lookup"
        << " (rank=" << moeEpParallelInfo.rank
        << ", ranks=" << moeEpParallelInfo.rankIds.size()
        << ", buffer=" << moeEpParallelInfo.bufferSize << ")";
    dispatchAndCombinecommDomain =
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
            moeEpParallelInfo.groupId,
            moeEpParallelInfo.rankIds,
            moeEpParallelInfo.rank,
            FLAGS_communication_backend,
            moeEpParallelInfo.bufferSize,
            /*streamId=*/0);
    dispatchAndCombineHcclComm =
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(
            dispatchAndCombinecommDomain);
  }
  if (dispatchAndCombineHcclComm == nullptr) {
    LOG(WARNING) << "CollectiveCommunicator still has no dispatch/combine HCCL "
                    "comm after fallback"
                 << " (comm_domain="
                 << (dispatchAndCombinecommDomain.empty()
                         ? "<empty>"
                         : dispatchAndCombinecommDomain)
                 << ")";
  }
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
  // Even when the global NPU backend is ATB, some models can still route
  // through native torch-style submodules that need ProcessGroup-based TP/DP
  // collectives. Keep the process groups available in ParallelArgs.
  int global_rank = parallel_args_->rank();
  int world_size = parallel_args_->world_size();
  int dp_size = parallel_args_->dp_size();
  int ep_size = parallel_args_->ep_size();

#if defined(USE_NPU)
  const int required_root_infos =
      get_collective_root_info_count(world_size, dp_size, ep_size);
  if (!root_infos_.empty()) {
    if (static_cast<int>(root_infos_.size()) < required_root_infos) {
      LOG(WARNING) << "CollectiveCommunicator received only "
                   << root_infos_.size() << " HCCL root infos, but needs "
                   << required_root_infos
                   << ". Falling back to torch ProcessGroup creation.";
    } else {
      LOG_FIRST_N(INFO, 1)
          << "CollectiveCommunicator creating NPU process groups from "
             "BRPC-synced HCCL root infos"
          << " (world_size=" << world_size << ", dp_size=" << dp_size
          << ", ep_size=" << ep_size << ", root_infos=" << root_infos_.size()
          << ")";

      process_group_ = create_process_group_from_root_info("world",
                                                           global_rank,
                                                           world_size,
                                                           world_size,
                                                           false,
                                                           device,
                                                           root_infos_[0]);
      parallel_args_->process_group_ = process_group_.get();

      const int tp_size = world_size / dp_size;
      tp_group_ = create_process_group_from_root_info(
          "tp",
          global_rank,
          world_size,
          tp_size,
          false,
          device,
          root_infos_[get_tp_root_info_index(
              global_rank, world_size, dp_size)]);
      parallel_args_->tp_group_ = tp_group_.get();

      if (dp_size > 1) {
        dp_local_process_group_ = create_process_group_from_root_info(
            "dp_local",
            global_rank,
            world_size,
            dp_size,
            true,
            device,
            root_infos_[get_dp_root_info_index(
                global_rank, world_size, dp_size)]);
        parallel_args_->dp_local_process_group_ = dp_local_process_group_.get();
      }

      if (ep_size > 1) {
        const int moe_tp_size = world_size / ep_size;
        moe_tp_group_ = create_process_group_from_root_info(
            "moe_tp",
            global_rank,
            world_size,
            moe_tp_size,
            false,
            device,
            root_infos_[get_moe_tp_root_info_index(
                global_rank, world_size, dp_size, ep_size)]);
        parallel_args_->moe_tp_group_ = moe_tp_group_.get();

        moe_ep_group_ = create_process_group_from_root_info(
            "moe_ep",
            global_rank,
            world_size,
            ep_size,
            true,
            device,
            root_infos_[get_moe_ep_root_info_index(
                global_rank, world_size, dp_size, ep_size)]);
        parallel_args_->moe_ep_group_ = moe_ep_group_.get();
      }
      return;
    }
  }
#endif

  std::string host;
  int port;
  net::parse_host_port_from_addr(master_addr, host, port);

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
  const int tp_group_port = port + port_offset;
  tp_group_ = create_process_group(global_rank,
                                   world_size,
                                   tp_size,
                                   tp_group_port,
                                   false,
                                   host,
                                   "tp_group_" + std::to_string(tp_group_port),
                                   device);
  parallel_args_->tp_group_ = tp_group_.get();
  // SP and TP share the same rank set during prefill today. Keep a distinct
  // handle so SP call sites do not depend on TP wiring directly.
  parallel_args_->sp_group_ = tp_group_.get();
  port += dp_size;

  if (dp_size > 1) {
    port_offset = global_rank % tp_size + 1;
    const int dp_group_port = port + port_offset;
    dp_local_process_group_ =
        create_process_group(global_rank,
                             world_size,
                             dp_size,
                             dp_group_port,
                             true,
                             host,
                             "dp_group_" + std::to_string(dp_group_port),
                             device);
    parallel_args_->dp_local_process_group_ = dp_local_process_group_.get();
    port += tp_size;
  }

  if (ep_size > 1) {
    int moe_tp_size = world_size / ep_size;
    port_offset = global_rank / moe_tp_size + 1;
    const int moe_tp_group_port = port + port_offset;
    moe_tp_group_ = create_process_group(
        global_rank,
        world_size,
        moe_tp_size,
        moe_tp_group_port,
        false,
        host,
        "moe_tp_group_" + std::to_string(moe_tp_group_port),
        device);
    parallel_args_->moe_tp_group_ = moe_tp_group_.get();
    port += ep_size;
    port_offset = global_rank % moe_tp_size + 1;
    const int moe_ep_group_port = port + port_offset;
    moe_ep_group_ = create_process_group(
        global_rank,
        world_size,
        ep_size,
        moe_ep_group_port,
        true,
        host,
        "moe_ep_group_" + std::to_string(moe_ep_group_port),
        device);
    parallel_args_->moe_ep_group_ = moe_ep_group_.get();
  }
}

const ParallelArgs* CollectiveCommunicator::parallel_args() {
  // TODO: init communicator
  return parallel_args_.get();
}

}  // namespace xllm
