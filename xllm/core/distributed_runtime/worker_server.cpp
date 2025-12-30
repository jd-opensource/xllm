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
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/collective_communicator.h"
#include "framework/parallel_state/mapping_npu.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/forward_params.h"
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

extern char** environ;

namespace xllm {
namespace {
void handle_signal(int signum) { _exit(0); }
}  // namespace

void WorkerServer::create_server(
    const runtime::Options& options,
    std::atomic<bool>& done,
    const std::string& master_node_addr,
    const torch::Device& d,
    int world_size,
    int global_rank,
    int32_t dp_size,
    int local_rank,
    int32_t ep_size,
    WorkerType worker_type,
    std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  Device device(d);
  device.set_device();
  LOG(INFO) << "Create worker server with device: " << device.index();

  auto worker_global_rank = global_rank;
  // TODO: FIXME Later
  // std::unique_ptr<WorkerImpl> worker_impl =
  // std::make_unique<LLMWorkerImpl>(...);
  // std::unique_ptr<WorkerServiceImpl> worker_service_impl =
  //    std::make_unique<WorkerServiceImpl>(worker_impl);
  auto worker_service = std::make_shared<WorkerService>(options, device);

  auto addr = net::get_local_ip_addr();
  auto worker_server =
      ServerRegistry::get_instance().register_server(server_name_);
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

  CollectiveCommunicator comm(worker_global_rank, world_size, dp_size, ep_size);
  const ParallelArgs* parallel_args = comm.parallel_args();
  comm.create_process_groups(master_node_addr, device);

  std::unique_ptr<Worker> worker =
      std::make_unique<Worker>(*parallel_args, device, options, worker_type);
  worker_service->set_worker(std::move(worker));
  if (options.enable_shm() && input_shm_manager && output_shm_manager) {
    worker_service->create_polling_shm_thread(std::move(input_shm_manager),
                                              std::move(output_shm_manager));
  }

  done.store(true);

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  worker_server->run();
}

void WorkerServer::create_spawn_server(int local_rank,
                                       const std::string& master_node_addr,
                                       std::atomic<bool>& done,
                                       const ParallelArgs& parallel_args,
                                       const torch::Device& d,
                                       const runtime::Options& options,
                                       WorkerType& worker_type) {
  auto local_rank_str = std::to_string(local_rank);
  const char* local_rank_ptr = local_rank_str.c_str();
  auto global_rank_str = std::to_string(parallel_args.rank());
  const char* global_rank_ptr = global_rank_str.c_str();
  auto world_size_str = std::to_string(parallel_args.world_size());
  const char* world_size_ptr = world_size_str.c_str();
  auto device_idx_str = std::to_string(d.index());
  const char* device_idx_ptr = device_idx_str.c_str();
  auto num_decoding_tokens_str = std::to_string(options.num_decoding_tokens());
  const char* num_decoding_tokens_ptr = num_decoding_tokens_str.c_str();
  auto block_size_str = std::to_string(options.block_size());
  const char* block_size_ptr = block_size_str.c_str();
  auto enable_shm_str = std::to_string(options.enable_shm());
  const char* enable_shm_ptr = enable_shm_str.c_str();
  auto is_local_str = std::to_string(options.is_local());
  const char* is_local_ptr = is_local_str.c_str();
  const char* worker_type_ptr = worker_type.to_string();
  std::string spawn_worker_bin_path =
      options.spawn_worker_path() + "/spawn_worker";
  LOG(INFO) << "Spawn worker path: " << spawn_worker_bin_path;
  const char* argv[] = {spawn_worker_bin_path.c_str(),
                        master_node_addr.c_str(),
                        local_rank_ptr,
                        global_rank_ptr,
                        world_size_ptr,
                        device_idx_ptr,
                        num_decoding_tokens_ptr,
                        block_size_ptr,
                        enable_shm_ptr,
                        is_local_ptr,
                        options.task_type().c_str(),
                        worker_type_ptr,
                        nullptr};
  pid_t pid;
  posix_spawn_file_actions_init(&file_actions_);
  posix_spawnattr_init(&spawn_attr_);
  int status = posix_spawnp(&pid,
                            argv[0],
                            &file_actions_,
                            &spawn_attr_,
                            const_cast<char**>(argv),
                            environ);
  if (status != 0) {
    LOG(ERROR) << "posix_spawnp failed: " << strerror(status);
    return;
  }
  use_spwan_worker_ = true;
  done.store(true);
}

void WorkerServer::prepare_shm(
    const ParallelArgs& parallel_args,
    const runtime::Options& options,
    std::unique_ptr<ForwardSharedMemoryManager>& input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager>& output_shm_manager) {
  if (options.is_local() && options.enable_shm()) {
    bool is_creator;
    int dp_local_tp_size = parallel_args.world_size() / parallel_args.dp_size();
    int dp_group = parallel_args.rank() / dp_local_tp_size;

    std::string name_prefix =
        "xllm_" + net::extract_port(options.master_node_addr().value());
    string name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, FORWARD_RAW_INPUT_TYPE, parallel_args.rank());
    input_shm_manager = std::make_unique<ForwardSharedMemoryManager>(
        name, PB_INPUT_SHM_SIZE, is_creator, FORWARD_RAW_INPUT_TYPE);
    LOG(INFO) << "Create input shared memory manager with name: " << name;

    name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, FORWARD_RAW_OUTPUT_TYPE, parallel_args.rank());
    output_shm_manager = std::make_unique<ForwardSharedMemoryManager>(
        name, PB_OUTPUT_SHM_SIZE, is_creator, FORWARD_RAW_OUTPUT_TYPE);
    LOG(INFO) << "Create output shared memory manager with name: " << name;
  }
}

WorkerServer::WorkerServer(int local_worker_idx,
                           const std::string& master_node_addr,
                           std::atomic<bool>& done,
                           const ParallelArgs& parallel_args,
                           const torch::Device& d,
                           const runtime::Options& options,
                           WorkerType worker_type,
                           bool use_spawn_worker)
    : server_name_("DistributeWorkerServer") {
  server_name_.append(std::to_string(options.server_idx()));

  if (worker_type == WorkerType::LLM || worker_type == WorkerType::ELM ||
      worker_type == WorkerType::VLM || worker_type == WorkerType::EVLM ||
      worker_type == WorkerType::REC) {
    if (use_spawn_worker) {
      // start worker in a spawn process(for offline inference worker.)
      create_spawn_server(local_worker_idx,
                          master_node_addr,
                          done,
                          parallel_args,
                          d,
                          options,
                          worker_type);
      return;
    } else {
      // worker process should handle SIGTREM and SIGINT signals.
      signal(SIGINT, handle_signal);
      signal(SIGTERM, handle_signal);

      std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager = nullptr;
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager = nullptr;
      prepare_shm(
          parallel_args, options, input_shm_manager, output_shm_manager);

      // start worker in a thread.
      worker_thread_ =
          std::make_unique<std::thread>(&WorkerServer::create_server,
                                        this,
                                        std::cref(options),
                                        std::ref(done),
                                        std::cref(master_node_addr),
                                        std::cref(d),
                                        parallel_args.world_size(),
                                        parallel_args.rank(),
                                        parallel_args.dp_size(),
                                        local_worker_idx,
                                        parallel_args.ep_size(),
                                        worker_type,
                                        std::move(input_shm_manager),
                                        std::move(output_shm_manager));
    }
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
  const int sleep_time_second = 3;
  while (try_count < FLAGS_max_reconnect_count) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, NULL);
    if (cntl.Failed()) {
      LOG(WARNING) << "Worker#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    } else {
      LOG(INFO) << "Worker#" << addr_info.global_rank() << " connect to "
                << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >= FLAGS_max_reconnect_count) {
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

  if (use_spwan_worker_) {
    posix_spawn_file_actions_destroy(&file_actions_);
    posix_spawnattr_destroy(&spawn_attr_);
  }

  auto worker_server = ServerRegistry::get_instance().get_server(server_name_);
  if (worker_server != nullptr) {
    worker_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

}  // namespace xllm
