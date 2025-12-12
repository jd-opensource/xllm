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

#pragma once

#include <string>

namespace xllm {

class SpawnWorkerServer final {
 public:
  explicit SpawnWorkerServer(const std::string& master_node_addr,
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
                             const std::string& task_type,
                             const std::string& worker_type);

  ~SpawnWorkerServer() = default;

  void run();

  static void handle_signal(int signum);

  static bool g_running_;
};

}  // namespace xllm
