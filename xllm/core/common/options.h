/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common/macros.h"
#include "common/types.h"

namespace xllm {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  std::string to_string() const;

 private:
  PROPERTY(std::string, model_path);

  PROPERTY(std::optional<std::string>, devices);

  PROPERTY(std::optional<std::string>, draft_model_path);

  PROPERTY(std::optional<std::string>, draft_devices);

  // max image num per prompt, default 4
  PROPERTY(int32_t, limit_image_per_prompt) = 4;

  // block size, default 128
  PROPERTY(int32_t, block_size) = 128;

  // the maximum cache size in bytes, default is 0 which means cache size is
  // caculated by available memory * max_memory_utilization
  PROPERTY(int64_t, max_cache_size) = 0;

  // maximum memory utilization allowed, default 0.9
  PROPERTY(double, max_memory_utilization) = 0.9;

  PROPERTY(bool, enable_prefix_cache) = true;

  // max tokens num per batch
  PROPERTY(int32_t, max_tokens_per_batch) = 20000;

  // max sequences num per batch
  PROPERTY(int32_t, max_seqs_per_batch) = 256;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill) = 512;

  // sps tokens
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  // thread num to handle requests
  PROPERTY(size_t, num_handling_threads) = 4;

  PROPERTY(std::optional<bool>, enable_eplb);

  PROPERTY(std::optional<int32_t>, redundant_experts_num);

  PROPERTY(std::optional<int64_t>, eplb_update_interval);

  PROPERTY(std::optional<double>, eplb_update_threshold);

  PROPERTY(std::optional<std::string>, communication_backend);

  PROPERTY(std::optional<std::string>, rank_tablefile);

  PROPERTY(std::optional<int32_t>, expert_parallel_degree);

  PROPERTY(std::string, task_type);

  PROPERTY(bool, enable_mla) = false;

  PROPERTY(bool, enable_chunked_prefill) = true;

  PROPERTY(std::optional<std::string>, master_node_addr);

  PROPERTY(int32_t, nnodes) = 1;

  PROPERTY(int32_t, node_rank) = 0;

  PROPERTY(int32_t, dp_size) = 1;

  PROPERTY(int32_t, ep_size) = 1;

  PROPERTY(std::optional<std::string>, xservice_addr);

  PROPERTY(std::optional<std::string>, instance_name);

  PROPERTY(bool, enable_disagg_pd) = false;

  PROPERTY(bool, enable_schedule_overlap) = true;

  PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(std::optional<std::string>, device_ip);

  PROPERTY(uint16_t, transfer_listen_port) = 26000;

  PROPERTY(std::optional<std::string>, etcd_addr);

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(std::optional<std::string>, tool_call_parser);

  PROPERTY(std::string, priority_strategy) = "FCFS";

  PROPERTY(bool, enable_online_preempt_offline) = true;

  PROPERTY(double, host_blocks_factor) = 0.0;

  PROPERTY(bool, enable_kvcache_store) = false;

  PROPERTY(bool, enable_cache_upload) = false;

  PROPERTY(std::string, store_protocol) = "tcp";

  PROPERTY(std::string, store_master_server_entry) = "";

  PROPERTY(std::string, store_metadata_connstring) = "";

  PROPERTY(bool, enable_continuous_kvcache) = false;
};

}  // namespace xllm
