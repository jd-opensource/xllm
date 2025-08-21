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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include <csignal>
#include <filesystem>
#include <memory>

#include "api_service/api_service.h"
#include "core/common/global_flags.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/runtime/master.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "server/xllm_server_registry.h"
using namespace xllm;

static std::atomic<uint32_t> signal_received{0};
void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str) {
  if (batch_sizes_str.empty() || batch_sizes_str == "auto") {
    return std::nullopt;
  }

  // parse devices string
  const std::vector<std::string> size_strs =
      absl::StrSplit(batch_sizes_str, ',');
  // remove duplicates
  std::unordered_set<uint32_t> sizes_set;
  for (const auto& size_str : size_strs) {
    uint32_t batch_size = 0;
    if (!absl::SimpleAtoi(size_str, &batch_size)) {
      LOG(ERROR) << "Failed to parse batch size: " << size_str;
      continue;
    }
    sizes_set.insert(batch_size);
  }

  if (sizes_set.empty()) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{sizes_set.begin(), sizes_set.end()};
}

int run() {
  // check if model path exists
  if (!std::filesystem::exists(FLAGS_model)) {
    LOG(FATAL) << "Model path " << FLAGS_model << " does not exist.";
  }

  if (FLAGS_model_id.empty()) {
    // use last part of the path as model id
    std::filesystem::path model_path =
        std::filesystem::path(FLAGS_model).lexically_normal();
    if (model_path.has_filename()) {
      FLAGS_model_id = std::filesystem::path(FLAGS_model).filename();
    } else {
      FLAGS_model_id =
          std::filesystem::path(FLAGS_model).parent_path().filename();
    }
  }

  if (FLAGS_host.empty()) {
    // set the host to the local IP when the host is empty
    FLAGS_host = net::get_local_ip_addr();
  }

  if (FLAGS_backend == "vlm") {
    FLAGS_enable_prefix_cache = false;
    FLAGS_enable_chunked_prefill = false;
  }

  // Create Master
  Options options;
  options.model_path(FLAGS_model)
      .task_type(FLAGS_task)
      .devices(FLAGS_devices)
      .draft_model_path(FLAGS_draft_model)
      .draft_devices(FLAGS_draft_devices)
      .limit_image_per_prompt(FLAGS_limit_image_per_prompt)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .max_tokens_per_batch(FLAGS_max_tokens_per_batch)
      .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
      .max_tokens_per_chunk_for_prefill(FLAGS_max_tokens_per_chunk_for_prefill)
      .num_speculative_tokens(FLAGS_num_speculative_tokens)
      .num_handling_threads(FLAGS_num_handling_threads)
      .communication_backend(FLAGS_communication_backend)
      .enable_eplb(FLAGS_enable_eplb)
      .redundant_experts_num(FLAGS_redundant_experts_num)
      .eplb_update_interval(FLAGS_eplb_update_interval)
      .eplb_update_threshold(FLAGS_eplb_update_threshold)
      .rank_tablefile(FLAGS_rank_tablefile)
      .expert_parallel_degree(FLAGS_expert_parallel_degree)
      .enable_mla(FLAGS_enable_mla)
      .enable_chunked_prefill(FLAGS_enable_chunked_prefill)
      .master_node_addr(FLAGS_master_node_addr)
      .instance_role(InstanceRole(FLAGS_instance_role))
      .device_ip(FLAGS_device_ip)
      .transfer_listen_port(FLAGS_transfer_listen_port)
      .nnodes(FLAGS_nnodes)
      .node_rank(FLAGS_node_rank)
      .dp_size(FLAGS_dp_size)
      .ep_size(FLAGS_ep_size)
      .xservice_addr(FLAGS_xservice_addr)
      .instance_name(FLAGS_host + ":" + std::to_string(FLAGS_port))
      .enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .etcd_addr(FLAGS_etcd_addr)
      .enable_service_routing(FLAGS_enable_service_routing)
      .tool_call_parser(FLAGS_tool_call_parser)
      .priority_strategy(FLAGS_priority_strategy)
      .enable_online_preempt_offline(FLAGS_enable_online_preempt_offline)
      .enable_cache_upload(FLAGS_enable_prefix_cache &&
                           FLAGS_enable_service_routing &&
                           FLAGS_enable_cache_upload)
      .host_blocks_factor(FLAGS_host_blocks_factor)
      .enable_kvcache_store(FLAGS_enable_kvcache_store &&
                            FLAGS_enable_prefix_cache &&
                            (FLAGS_host_blocks_factor > 0.0))
      .store_protocol(FLAGS_store_protocol)
      .store_master_server_entry(FLAGS_store_master_server_entry)
      .store_metadata_connstring(FLAGS_store_metadata_connstring)
      .enable_continuous_kvcache(FLAGS_enable_continuous_kvcache);

  InstanceName::name()->set_name(options.instance_name().value_or(""));

  // working node
  if (options.node_rank() != 0) {
    auto master = std::make_unique<LLMAssistantMaster>(options);
    master->run();
    return 0;
  }

  // master node
  auto master = create_master(FLAGS_backend, options);
  master->run();

  // supported models
  std::vector<std::string> model_names = {FLAGS_model_id};
  std::filesystem::path model_path =
      std::filesystem::path(FLAGS_model).lexically_normal();
  std::string model_version;
  if (model_path.has_filename()) {
    model_version = std::filesystem::path(FLAGS_model).filename();
  } else {
    model_version = std::filesystem::path(FLAGS_model).parent_path().filename();
  }
  std::vector<std::string> model_versions = {model_version};

  auto api_service =
      std::make_unique<APIService>(master.get(), model_names, model_versions);
  auto xllm_server =
      ServerRegistry::get_instance().register_server("HttpServer");

  // start brpc server
  if (!xllm_server->start(std::move(api_service))) {
    LOG(ERROR) << "Failed to start brpc server on port " << FLAGS_port;
    return -1;
  }

  return 0;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  google::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging("xllm");

  return run();
}
