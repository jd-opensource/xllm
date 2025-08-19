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

  // block size, default 16
  PROPERTY(int32_t, block_size) = 16;

  // the maximum cache size in bytes, default is 0 which means cache size is
  // caculated by available memory * max_memory_utilization
  PROPERTY(int64_t, max_cache_size) = 0;

  // maximum memory utilization allowed, default 0.9
  PROPERTY(double, max_memory_utilization) = 0.9;

  PROPERTY(bool, enable_prefix_cache) = true;

  // max tokens num per batch
  PROPERTY(int32_t, max_tokens_per_batch) = std::numeric_limits<int32_t>::max();

  // max sequences num per batch
  PROPERTY(int32_t, max_seqs_per_batch) = 256;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill) = 2048;

  // sps tokens
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  // thread num to handle requests
  PROPERTY(size_t, num_handling_threads) = 4;

  PROPERTY(std::optional<bool>, enable_eplb);

  PROPERTY(std::optional<int64_t>, eplb_update_rate);

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

  PROPERTY(bool, enable_schedule_overlap) = false;

  PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(std::optional<std::string>, device_ip);

  PROPERTY(uint16_t, transfer_listen_port) = 26000;

  PROPERTY(std::optional<std::string>, etcd_addr);

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(std::optional<std::string>, tool_call_parser);
};

}  // namespace xllm
