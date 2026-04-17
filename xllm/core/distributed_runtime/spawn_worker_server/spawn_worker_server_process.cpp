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

#include <cstdlib>

#include "spawn_worker_server.h"

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @num_decoding_tokens
// @block_size
// @enable_prefix_cache
// @enable_chunked_prefill
// @enable_schedule_overlap
// @enable_shm
// @use_contiguous_input_buffer
// @enable_graph
// @enable_graph_mode_decode_no_padding
// @enable_prefill_piecewise_graph
// @enable_block_copy_kernel
// @enable_beam_search_kernel
// @enable_rec_fast_sampler
// @is_local
// @enable_prefill_sp
// @task_type
// @worker_type
// @input_shm_size
// @output_shm_size
// @communication_backend
int main(int argc, char* argv[]) {
  if (argc < 26) {
    LOG(ERROR)
        << "Spawn worker process receive wrong args. Need 26 args, receive "
        << argc;
    return 1;
  }

  std::string master_node_addr = std::string(argv[1]);
  int local_rank = atoi(argv[2]);
  int global_rank = atoi(argv[3]);
  int world_size = atoi(argv[4]);
  int device_idx = atoi(argv[5]);
  int num_decoding_tokens = atoi(argv[6]);
  int block_size = atoi(argv[7]);
  int enable_prefix_cache = atoi(argv[8]);
  int enable_chunked_prefill = atoi(argv[9]);
  int enable_schedule_overlap = atoi(argv[10]);
  int enable_shm = atoi(argv[11]);
  int use_contiguous_input_buffer = atoi(argv[12]);
  int enable_graph = atoi(argv[13]);
  int enable_graph_mode_decode_no_padding = atoi(argv[14]);
  int enable_prefill_piecewise_graph = atoi(argv[15]);
  int enable_block_copy_kernel = atoi(argv[16]);
  int enable_beam_search_kernel = atoi(argv[17]);
  int enable_rec_fast_sampler = atoi(argv[18]);
  int is_local = atoi(argv[19]);
  int enable_prefill_sp = atoi(argv[20]);
  std::string task_type = std::string(argv[21]);
  std::string worker_type = std::string(argv[22]);
  uint64_t input_shm_size = atoll(argv[23]);
  uint64_t output_shm_size = atoll(argv[24]);
  std::string communication_backend = std::string(argv[25]);

  LOG(INFO) << "Spawn worker: "
            << "master_node_addr = " << master_node_addr
            << ", local_rank = " << local_rank
            << ", world_size = " << world_size
            << ", device_idx = " << device_idx
            << ", num_decoding_tokens = " << num_decoding_tokens
            << ", block_size = " << block_size
            << ", enable_prefix_cache = " << (enable_prefix_cache > 0)
            << ", enable_chunked_prefill = " << (enable_chunked_prefill > 0)
            << ", enable_schedule_overlap = " << (enable_schedule_overlap > 0)
            << ", enable_shm = " << (enable_shm > 0)
            << ", use_contiguous_input_buffer = "
            << (use_contiguous_input_buffer > 0)
            << ", enable_graph = " << (enable_graph > 0)
            << ", enable_graph_mode_decode_no_padding = "
            << (enable_graph_mode_decode_no_padding > 0)
            << ", enable_prefill_piecewise_graph = "
            << (enable_prefill_piecewise_graph > 0)
            << ", enable_block_copy_kernel = " << (enable_block_copy_kernel > 0)
            << ", enable_beam_search_kernel = "
            << (enable_beam_search_kernel > 0)
            << ", enable_rec_fast_sampler = " << (enable_rec_fast_sampler > 0)
            << ", input_shm_size = " << input_shm_size
            << ", output_shm_size = " << output_shm_size
            << ", is_local = " << (is_local > 0)
            << ", enable_prefill_sp = " << (enable_prefill_sp > 0)
            << ", task_type = " << task_type
            << ", worker_type = " << worker_type
            << ", communication_backend = " << communication_backend << "\n";

  xllm::SpawnWorkerServer worker(master_node_addr,
                                 local_rank,
                                 global_rank,
                                 world_size,
                                 device_idx,
                                 num_decoding_tokens,
                                 block_size,
                                 enable_prefix_cache > 0,
                                 enable_chunked_prefill > 0,
                                 enable_schedule_overlap > 0,
                                 enable_shm > 0,
                                 use_contiguous_input_buffer > 0,
                                 enable_graph > 0,
                                 enable_graph_mode_decode_no_padding > 0,
                                 enable_prefill_piecewise_graph > 0,
                                 enable_block_copy_kernel > 0,
                                 enable_beam_search_kernel > 0,
                                 enable_rec_fast_sampler > 0,
                                 input_shm_size,
                                 output_shm_size,
                                 is_local > 0,
                                 enable_prefill_sp > 0,
                                 task_type,
                                 worker_type,
                                 communication_backend);

  worker.run();

  return 0;
}
