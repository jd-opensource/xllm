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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/prctl.h>

#include "spawn_worker_server.h"

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @num_decoding_tokens
// @block_size
int main(int argc, char* argv[]) {
  if (argc < 7) {
    LOG(ERROR)
        << "Spwan worker process receive wrong args. Need 7 args, receive "
        << argc;
    return 1;
  }

  // set PR_SET_PDEATHSIG flag that child should exit
  // when parent process exit
  if (prctl(PR_SET_PDEATHSIG, SIGHUP) == -1) {
    perror("prctl");
    return EXIT_FAILURE;
  }

  std::string master_node_addr = std::string(argv[1]);
  int local_rank = atoi(argv[2]);
  int global_rank = atoi(argv[3]);
  int world_size = atoi(argv[4]);
  int device_idx = atoi(argv[5]);
  int num_decoding_tokens = atoi(argv[6]);
  int block_size = atoi(argv[7]);

  LOG(INFO) << "Spwan worker: "
            << "master_node_addr = " << master_node_addr
            << ", local_rank = " << local_rank
            << ", world_size = " << world_size
            << ", device_idx = " << device_idx
            << ", num_decoding_tokens = " << num_decoding_tokens
            << ", block_size = " << block_size << "\n";

  xllm::SpawnWorkerServer worker(master_node_addr,
                                 local_rank,
                                 global_rank,
                                 world_size,
                                 device_idx,
                                 num_decoding_tokens,
                                 block_size);

  worker.run();

  return 0;
}
