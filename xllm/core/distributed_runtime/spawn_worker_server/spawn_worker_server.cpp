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

#include <absl/strings/str_split.h>
#if defined(USE_NPU)
#include <acl/acl.h>
#endif
#include <signal.h>
#include <sys/prctl.h>

#include <cstdlib>

#include "core/distributed_runtime/worker_server.h"
#include "core/platform/device.h"
#include "core/runtime/options.h"

namespace xllm {

bool xllm::SpawnWorkerServer::g_running_ = true;

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

  ParallelArgs parallel_args(global_rank, world_size, 1, nullptr, 1);
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

void SpawnWorkerServer::handle_signal(int signum) { g_running_ = false; }

void SpawnWorkerServer::run() {
  signal(SIGINT, SpawnWorkerServer::handle_signal);
  signal(SIGTERM, SpawnWorkerServer::handle_signal);

  // main thread waiting here
  while (SpawnWorkerServer::g_running_) {
    sleep(5);
  }
}

}  // namespace xllm
