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

#include "worker_server.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/mapping_npu.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"
#include "util/threadpool.h"
#include "util/timer.h"
#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "xllm_kernels/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_kernels/core/include/atb_speed/utils/singleton.h"
#include "xllm_kernels/models/base/param/mapping.h"
#endif

namespace xllm {

void WorkerServer::create_server(const runtime::Options& options,
                                 std::atomic<bool>& done,
                                 const std::string& master_node_addr,
                                 const torch::Device& device,
                                 int world_size,
                                 int global_rank,
                                 int32_t dp_size,
                                 int local_rank,
                                 int32_t ep_size) {
  int device_id = device.index();
#if defined(USE_NPU)
  int ret = aclrtSetDevice(device_id);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id: " << device_id << " failed, ret:" << ret;
  }
#elif defined(USE_MLU)
// TODO(mlu): implement mlu device set
#endif

  auto worker_global_rank = global_rank;
  // TODO: FIXME Later
  // std::unique_ptr<WorkerImpl> worker_impl =
  // std::make_unique<LLMWorkerImpl>(...);
  // std::unique_ptr<WorkerServiceImpl> worker_service_impl =
  //    std::make_unique<WorkerServiceImpl>(worker_impl);
  auto worker_service = std::make_shared<WorkerService>(options, device);

  auto addr = net::get_local_ip_addr();
  auto worker_server =
      ServerRegistry::get_instance().register_server("DistributeWorkerServer");
  if (!worker_server->start(worker_service, addr + ":0")) {
    LOG(ERROR) << "failed to start distribute worker server on address: "
               << addr;
    return;
  }

  auto worker_server_addr =
      addr + ":" + std::to_string(worker_server->listen_port());
  LOG(INFO) << "Worker " << worker_global_rank
            << ": server address: " << worker_server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(worker_server_addr);
  addr_info.set_global_rank(worker_global_rank);
  proto::CommUniqueIdList uids;
  sync_master_node(master_node_addr, addr_info, uids);

  auto dp_local_process_group_num =
      (dp_size > 1 && dp_size < world_size) ? dp_size : 0;
  std::unique_ptr<ParallelArgs> parallel_args;
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
  //     world_size, &unique_ids[0], worker_global_rank, &comm);
  // CHECK(hccl_result == HCCL_SUCCESS)
  //     << "HcclCommInitRootInfo failed, global rank is " <<
  //     worker_global_rank;
  // std::unique_ptr<ProcessGroupHCCL> hccl_pg =
  //     std::make_unique<ProcessGroupHCCL>(
  //         worker_global_rank, world_size, device, comm);
  MappingNPU::Options mapping_options;
  mapping_options.dp_size(dp_size)
      .tp_size(world_size / dp_size)
      .moe_tp_size(world_size / ep_size)
      .moe_ep_size(ep_size)
      .pp_size(1)
      .sp_size(1);

  MappingNPU mapping_npu(
      FLAGS_rank_tablefile, world_size, worker_global_rank, mapping_options);

  auto mapping_data = mapping_npu.to_json();
#if defined(USE_NPU)
  atb_speed::base::Mapping mapping;
  mapping.ParseParam(mapping_data);
#if defined(USE_A3)
  mapping.InitGlobalCommDomain(FLAGS_communication_backend);
#else
  mapping.InitCommDomain(FLAGS_communication_backend);
#endif
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
  parallel_args = std::make_unique<ParallelArgs>(worker_global_rank,
                                                 world_size,
                                                 dp_size,
                                                 nullptr,
                                                 ep_size,
                                                 mapping_data,
                                                 mapping,
                                                 dispatchAndCombinecommDomain,
                                                 dispatchAndCombineHcclComm);
#elif defined(USE_MLU)
  parallel_args = std::make_unique<ParallelArgs>(
      worker_global_rank, world_size, dp_size, nullptr, ep_size);
#endif

  WorkerType worker_type =
      (options.task_type() == "generate") ? WorkerType::LLM : WorkerType::ELM;
  CHECK(worker_type == WorkerType::LLM || worker_type == WorkerType::ELM)
      << "Multi Node only support LLM and ELM Now, but get task type = "
      << options.task_type();
  std::unique_ptr<Worker> worker =
      std::make_unique<Worker>(*parallel_args, device, options, worker_type);
  worker_service->set_worker(std::move(worker));
  done.store(true);

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  worker_server->run();
}

WorkerServer::WorkerServer(int local_worker_idx,
                           const std::string& master_node_addr,
                           std::atomic<bool>& done,
                           const ParallelArgs& parallel_args,
                           const torch::Device& device,
                           const runtime::Options& options,
                           WorkerType worker_type) {
  if (worker_type == WorkerType::LLM || worker_type == WorkerType::ELM) {
    // TODO: Use Process or thread.
    worker_thread_ = std::make_unique<std::thread>(&WorkerServer::create_server,
                                                   this,
                                                   std::cref(options),
                                                   std::ref(done),
                                                   std::cref(master_node_addr),
                                                   std::cref(device),
                                                   parallel_args.world_size(),
                                                   parallel_args.rank(),
                                                   parallel_args.dp_size(),
                                                   local_worker_idx,
                                                   parallel_args.ep_size());
  } else {
    // TODO: support other model type later.
    LOG(ERROR) << "Unsupported model type: " << worker_type;
  }
}

bool WorkerServer::sync_master_node(const std::string& master_node_addr,
                                    proto::AddressInfo& addr_info,
                                    proto::CommUniqueIdList& uids) {
  // Brpc connection resources
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = 10000;
  options.max_retry = 3;
  if (channel.Init(master_node_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Failed to initialize BRPC channel to " << master_node_addr;
    return false;
  }
  proto::Collective_Stub stub(&channel);

  // Retry until master node ready
  int try_count = 0;
  brpc::Controller cntl;
  while (try_count < FLAGS_max_connect_count) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, NULL);
    if (cntl.Failed()) {
      LOG(WARNING) << "Worker#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_sleep_time_second));
    } else {
      LOG(INFO) << "Worker#" << addr_info.global_rank() << " connect to "
                << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "Worker#" << addr_info.global_rank() << " connect to "
               << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

WorkerServer::~WorkerServer() {
  if (worker_thread_->joinable()) {
    worker_thread_->join();
  }
}

}  // namespace xllm
