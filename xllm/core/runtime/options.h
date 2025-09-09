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

#include <torch/torch.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.h"
#include "common/types.h"

namespace xllm {
namespace runtime {

struct Options {
  PROPERTY(std::string, model_path);

  PROPERTY(std::optional<std::string>, draft_model_path);

  // devices for execute model
  PROPERTY(std::vector<torch::Device>, devices);

  // devices for execute draft model
  PROPERTY(std::vector<torch::Device>, draft_devices);

  // the number of slots per block, default 128, value must be multiple of 16
  PROPERTY(int32_t, block_size) = 128;

  // 0 means that cache size is caculated by available memory
  PROPERTY(int64_t, max_cache_size) = 0;

  // maximum memory utilization allowed, default 0.9
  PROPERTY(double, max_memory_utilization) = 0.9;

  // enable prefix cache
  PROPERTY(bool, enable_prefix_cache) = true;

  // number of decoding tokens per sequence
  // in speculative decoding, it is the number of speculative tokens + 1
  PROPERTY(int64_t, num_decoding_tokens) = 1;

  // the number of speculative tokens per step
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  // enable speculative decode
  PROPERTY(bool, enable_speculative_decode) = false;

  PROPERTY(int32_t, world_size) = 1;

  // task type, support 'generate' and 'embed' currently
  PROPERTY(std::string, task_type) = "generate";

  PROPERTY(bool, enable_mla) = false;

  // master node address when we launch a multi-node task.
  PROPERTY(std::optional<std::string>, master_node_addr);

  // total nodes num
  PROPERTY(int32_t, nnodes) = 1;

  // the node_rank of current worker process at.
  PROPERTY(int32_t, node_rank) = 0;

  // data parallelism size, currently mainly used for MoE model
  // default set as 1 for non-MoE model
  PROPERTY(int32_t, dp_size) = 1;

  // expert parallelism size, currently mainly used for MoE model
  // Default set as 1 for non-MoE model.
  PROPERTY(int32_t, ep_size) = 1;

  // enable enable_schedule_overlap to improve runtime execution efficiency.
  PROPERTY(bool, enable_schedule_overlap) = true;

  // enable chunked prefill.
  PROPERTY(bool, enable_chunked_prefill) = true;

  // the max sequences limit of a batch.
  PROPERTY(int32_t, max_seqs_per_batch) = 256;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill) = 512;

  // for master service, master server addr
  PROPERTY(std::optional<std::string>, xservice_addr);

  // for master service, current instance name(ID).
  PROPERTY(std::optional<std::string>, instance_name);

  // enable disaggregated prefill-decode mode.
  PROPERTY(bool, enable_disagg_pd) = false;

  // instance role, support `DEFAULT`, `PREFILL`, `DECODE`
  PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;

  // transfer kv mode in disaggregated prefill and decode execution.
  // support `PUSH` and `PULL`
  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  // device_ip needed in disaggregated prefill and decode execution.
  PROPERTY(std::optional<std::string>, device_ip);

  // transfer_listen_port needed in disaggregated prefill and decode execution.
  PROPERTY(uint16_t, transfer_listen_port) = 26000;

  // enable service routing mode.
  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(std::string, priority_strategy) = "FCFS";

  PROPERTY(bool, enable_online_preempt_offline) = true;

  // enable kvcache upload to service.
  PROPERTY(bool, enable_cache_upload) = false;

  // host block factor, e.g. host block num = host_blocks_factor * hbm block num
  PROPERTY(double, host_blocks_factor) = 0.0;

  // enable kvcache store.
  PROPERTY(bool, enable_kvcache_store) = false;

  // store transfer protocol.
  PROPERTY(std::string, store_protocol) = "tcp";

  // The address information of the Master (IP:Port for default mode and
  // etcd://IP:Port;IP:Port;...;IP:Port for high availability mode)
  PROPERTY(std::string, store_master_server_entry) = "";

  // the address of the metadata service (e.g., etcd/Redis) required for
  // Transfer Engine initialization
  PROPERTY(std::string, store_metadata_connstring) = "";

  // enable continuous kvcache
  PROPERTY(bool, enable_continuous_kvcache) = false;
};

}  // namespace runtime
}  // namespace xllm
