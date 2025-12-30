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

#include "distributed_runtime/dist_manager.h"

#include <glog/logging.h>

#include "comm_channel.h"
#include "distributed_runtime/collective_service.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/llm_worker_impl.h"
#include "server/xllm_server_registry.h"
#include "shm_channel.h"
#include "util/net.h"
namespace xllm {

DistManager::DistManager(const runtime::Options& options)
    : server_name_("CollectiveServer") {
  auto master_node_addr = options.master_node_addr().value_or("");
  if (!master_node_addr.empty()) {
    server_name_.append(std::to_string(options.server_idx()));
    setup_multi_node_workers(options, master_node_addr);
  } else {
    LOG(FATAL) << "master_node_addr is empty.";
  }
}

DistManager::~DistManager() {
  XllmServer* collective_server =
      ServerRegistry::get_instance().get_server(server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

namespace {
std::unique_ptr<CommChannel> create_channel(const std::string& worker_addrs,
                                            int r,
                                            int dp_local_tp_size,
                                            const runtime::Options& options) {
  std::unique_ptr<CommChannel> channel;

  if (net::extract_ip(options.master_node_addr().value_or("")) ==
          net::extract_ip(worker_addrs) &&
      options.enable_shm()) {
    // create shared memory manager for local rank
    bool is_driver = false;
    int dp_group = r / dp_local_tp_size;
    if (r % dp_local_tp_size == 0) {
      is_driver = true;
    }
    channel = std::make_unique<ShmChannel>(dp_group, r, is_driver, options);
  } else {
    channel = std::make_unique<CommChannel>();
  }

  channel->init_brpc(worker_addrs);

  return channel;
}

}  // namespace

void DistManager::setup_multi_node_workers(
    const runtime::Options& options,
    const std::string& master_node_addr) {
  const auto& devices = options.devices();

  // Process/Thread Worker Mode, we use it in multi-nodes serving.

  // Here, we assume that all node use same index devices. That is, if we set
  // device='1,2,3,4' and nnodes=2, then both machine nodes will use the
  // devices '1,2,3,4'. Therefore, the total world size is 2 * 4 = 8. This
  // means that each of the two nodes will utilize four devices (specifically
  // devices 1, 2, 3, and 4), resulting in a total of 8 devices being used
  // across the entire distributed setup.

  // To maintain interface consistency, we have implemented a new WorkerImpl
  // class. In this class, we create processes, initialize NCCL ProcessGroup,
  // set up GRPC servers, and so on.

  std::vector<std::atomic<bool>> dones(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    dones[i].store(false, std::memory_order_relaxed);
  }

  CHECK_GE(options.nnodes(), 1) << "At least one node is required";
  CHECK_GE(options.node_rank(), 0) << "Node rank must >= 0.";
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  const int32_t world_size = each_node_ranks * options.nnodes();
  const int32_t base_rank = options.node_rank() * each_node_ranks;
  const int32_t dp_size = options.dp_size();
  const int32_t ep_size = options.ep_size();
  const int32_t dp_local_tp_size = world_size / dp_size;

  LOG(INFO) << "Multi-node serving world_size = " << world_size
            << ", each_node_ranks = " << each_node_ranks
            << ", current node rank = " << options.node_rank()
            << ", nnodes = " << options.nnodes() << ", dp_size = " << dp_size
            << ", ep_size = " << ep_size << ", tp_size = " << dp_local_tp_size;

  CHECK_EQ((world_size % dp_size), 0)
      << "Global world size must be divisible by dp size in multi-node "
         "serving mode.";

  runtime::Options worker_server_options = options;
  worker_server_options.world_size(world_size);
  WorkerType worker_type("LLM");
  const auto& model_backend = options.backend();
  if (model_backend == "llm") {
    worker_type =
        (options.task_type() == "generate") ? WorkerType::LLM : WorkerType::ELM;
  } else if (model_backend == "vlm") {
    worker_type = (options.task_type() == "generate") ? WorkerType::VLM
                                                      : WorkerType::EVLM;
  } else if (model_backend == "rec") {
    worker_type = WorkerType::REC;
  } else {
    LOG(ERROR) << "Unsupported " << model_backend << " in multi-node.";
  }
  // create local workers
  for (size_t i = 0; i < devices.size(); ++i) {
    // worldsize = 8
    // Node1: 0, 1, 2, 3
    // Node2: 0+4, 1+4, 2+4, 3+4
    const int32_t rank = static_cast<int32_t>(i) + base_rank;

    // we use spawn process worker to launch a xllm instance
    // when start a offline inference task with multi-gpu/npu/mpu/...
    bool use_spawn_worker = options.enable_offline_inference() && i > 0;
    ParallelArgs parallel_args(rank, world_size, dp_size, nullptr, ep_size);

    servers_.emplace_back(std::make_unique<WorkerServer>(i,
                                                         master_node_addr,
                                                         // done,
                                                         dones[i],
                                                         parallel_args,
                                                         devices[i],
                                                         worker_server_options,
                                                         worker_type,
                                                         use_spawn_worker));
  }

  // Master node need to wait all workers done
  if (options.node_rank() == 0) {
    // if dp_size equals 1, use global process group directly
    // if dp_size equals world_size, distributed communication is not required
    auto dp_local_process_group_num =
        (dp_size > 1 && dp_size < world_size) ? dp_size : 0;

    // create collective server to sync all workers.
    std::shared_ptr<CollectiveService> collective_service =
        std::make_shared<CollectiveService>(
            dp_local_process_group_num, world_size, devices[0].index());
    XllmServer* collective_server =
        ServerRegistry::get_instance().register_server(server_name_);
    if (!collective_server->start(collective_service, master_node_addr)) {
      LOG(ERROR) << "failed to start collective server on address: "
                 << master_node_addr;
      return;
    }

    auto worker_addrs_map = collective_service->wait();

    // check if all workers connected
    // and then create worker clients
    for (size_t r = 0; r < world_size; ++r) {
      if (worker_addrs_map.find(r) == worker_addrs_map.end()) {
        LOG(FATAL) << "Not all worker connect to engine server. Miss rank is "
                   << r;
        return;
      }
      auto channel =
          create_channel(worker_addrs_map[r], r, dp_local_tp_size, options);
      worker_clients_.emplace_back(
          std::make_unique<RemoteWorker>(r,
                                         worker_addrs_map[r],
                                         devices[r % each_node_ranks],
                                         std::move(channel)));
    }
  }

  for (int idx = 0; idx < dones.size(); ++idx) {
    while (!dones[idx].load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}
}  // namespace xllm
