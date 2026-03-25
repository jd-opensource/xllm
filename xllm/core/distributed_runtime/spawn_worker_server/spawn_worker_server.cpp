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

#include "spawn_worker_server.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#endif
#include <signal.h>
#include <unistd.h>

#include "core/distributed_runtime/worker_server.h"
#include "core/platform/device.h"
#if defined(USE_CUDA)
#include "core/platform/numa_utils.h"
#endif
#include "core/runtime/options.h"

namespace xllm {

namespace {
std::string get_backend_from_worker_type(const std::string& worker_type) {
  if (worker_type == "LLM" || worker_type == "ELM") {
    return "llm";
  }
  if (worker_type == "VLM" || worker_type == "EVLM" ||
      worker_type == "MMEVLM") {
    return "vlm";
  }
  if (worker_type == "REC") {
    return "rec";
  }
  if (worker_type == "DIT") {
    return "dit";
  }
  return "";
}
}  // namespace

SpawnWorkerServer::SpawnWorkerServer(const std::string& master_node_addr,
                                     int local_rank,
                                     int global_rank,
                                     int world_size,
                                     int device_idx,
                                     int num_decoding_tokens,
                                     int block_size,
                                     bool enable_shm,
                                     uint64_t input_shm_size,
                                     uint64_t output_shm_size,
                                     bool is_local,
                                     bool enable_prefill_sp,
                                     const std::string& task_type,
                                     const std::string& worker_type,
                                     const std::string& communication_backend) {
  // TODO: pass whole xllm::runtime::Options here from main process.
  xllm::runtime::Options runner_options;
  const std::string backend = get_backend_from_worker_type(worker_type);
  CHECK(!backend.empty()) << "Unsupported worker_type for backend mapping: "
                          << worker_type;
  runner_options.block_size(block_size)
      .backend(backend)
      .num_decoding_tokens(num_decoding_tokens)
      .enable_prefill_sp(enable_prefill_sp)
      .enable_schedule_overlap(false)
      .enable_offline_inference(true)
      .master_node_addr(master_node_addr)
      .enable_shm(enable_shm)
      .input_shm_size(input_shm_size)
      .output_shm_size(output_shm_size)
      .is_local(is_local)
      .task_type(task_type);
  FLAGS_enable_schedule_overlap = false;
  FLAGS_enable_prefill_sp = enable_prefill_sp;
  FLAGS_master_node_addr = master_node_addr;
  FLAGS_block_size = block_size;
  FLAGS_communication_backend = communication_backend;

  const std::string device_type = xllm::Device::type_str();
  const std::string device_str = device_type + ":" + std::to_string(device_idx);
  xllm::Device device{torch::Device(device_str)};
  device.set_device();

#if defined(USE_NPU)
  device.init_device_context();
  FLAGS_enable_atb_comm_multiprocess = true;
#endif

#if defined(USE_CUDA)
  // Bind worker process to the same NUMA node as the device
  // This prevents the process from spanning across NUMA nodes, which would
  // significantly degrade memory access and other performance aspects
  int32_t numa_node = numa::get_device_numa_node(device_idx);
  if (numa_node >= 0) {
    LOG(INFO) << "Worker process (device " << device_idx
              << ") binding to NUMA node " << numa_node;
    int32_t ret = numa::bind_process_to_numa_node(numa_node);
    if (ret != 0) {
      LOG(WARNING) << "Failed to bind worker process to NUMA node " << numa_node
                   << ", continuing without NUMA binding";
    }
  } else {
    LOG(INFO) << "NUMA node detection not available or not needed for device "
              << device_idx;
  }
#endif

  ParallelArgs parallel_args(global_rank,
                             world_size,
                             /* dp_size = */ 1,
                             /*cp_size = */ 1,
                             /*process_group = */ nullptr,
                             /*ep_size = */ 1);
  worker_server_ = std::make_unique<WorkerServer>(local_rank,
                                                  master_node_addr,
                                                  done_,
                                                  parallel_args,
                                                  device,
                                                  runner_options,
                                                  worker_type,
                                                  false);
}

SpawnWorkerServer::~SpawnWorkerServer() = default;

void SpawnWorkerServer::handle_signal(int signum) {
  (void)signum;
  _exit(0);
}

void SpawnWorkerServer::run() {
  signal(SIGINT, SpawnWorkerServer::handle_signal);
  signal(SIGTERM, SpawnWorkerServer::handle_signal);
  signal(SIGHUP, SpawnWorkerServer::handle_signal);

  // Keep process alive until SIGTERM/SIGINT arrives from parent teardown.
  while (true) {
    pause();
  }
}

}  // namespace xllm
