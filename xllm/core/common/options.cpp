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

#include "options.h"

namespace xllm {
std::string Options::to_string() const {
  std::stringstream ss;
  ss << "Options: [";
  ss << "model_path: " << model_path()
     << ", devices: " << devices().value_or("null")
     << ", draft_model_path: " << draft_model_path().value_or("null")
     << ", draft_devices: " << draft_devices().value_or("null")
     << ",limit_image_per_prompt: " << limit_image_per_prompt()
     << ", block_size: " << block_size()
     << ", max_cache_size: " << max_cache_size()
     << ", max_memory_utilization: " << max_memory_utilization()
     << ", enable_prefix_cache: " << enable_prefix_cache()
     << ", max_tokens_per_batch: " << max_tokens_per_batch()
     << ", max_seqs_per_batch: " << max_seqs_per_batch()
     << ", max_tokens_per_chunk_for_prefill: "
     << max_tokens_per_chunk_for_prefill()
     << ", num_speculative_tokens: " << num_speculative_tokens()
     << ", num_handling_threads: " << num_handling_threads()
     << ", communication_backend: " << communication_backend().value_or("null")
     << ", rank_tablefile: " << rank_tablefile().value_or("null")
     << ", expert_parallel_degree: " << expert_parallel_degree().value_or(0)
     << ", task_type: " << task_type() << ", enable_mla: " << enable_mla()
     << ", enable_chunked_prefill: " << enable_chunked_prefill()
     << ", master_node_addr: " << master_node_addr().value_or("null")
     << ", instance_role: " << instance_role().to_string()
     << ", device_ip: " << device_ip().value_or("null")
     << ", transfer_listen_port: " << transfer_listen_port()
     << ", nnodes: " << nnodes() << ", node_rank: " << node_rank()
     << ", enable_schedule_overlap: " << enable_schedule_overlap()
     << ", enable_disagg_pd: " << enable_disagg_pd()
     << ", kv_cache_transfer_mode: " << kv_cache_transfer_mode()
     << ", etcd_addr: " << etcd_addr().value_or("null")
     << ", enable_service_routing: " << enable_service_routing()
     << ", enable_cache_upload: " << enable_cache_upload()
     << ", enable_kvcache_store: " << enable_kvcache_store()
     << ", store_protocol: " << store_protocol()
     << ", store_master_server_entry: " << store_master_server_entry()
     << ", store_metadata_connstring: " << store_metadata_connstring()
     << ", enable_continuous_kvcache: " << enable_continuous_kvcache();
  ss << "]";
  return ss.str();
}

}  // namespace xllm
