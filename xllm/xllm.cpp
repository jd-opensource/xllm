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
#include "core/common/help_formatter.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/master.h"
#include "core/util/json_reader.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "models/model_registry.h"
#include "server/xllm_server_registry.h"

using namespace xllm;

static std::atomic<uint32_t> signal_received{0};

static std::unordered_set<std::string> deepseek_like_model_set = {
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
    "deepseek_mtp",
    "kimi_k2",
};

void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

std::string get_model_type(const std::filesystem::path& model_path) {
  JsonReader reader;
  // for llm, vlm and rec models, the config.json file is in the model path
  std::filesystem::path config_json_path = model_path / "config.json";

  if (std::filesystem::exists(config_json_path)) {
    reader.parse(config_json_path);
    std::string model_type = reader.value<std::string>("model_type").value();
    if (model_type.empty()) {
      LOG(FATAL) << "Please check config.json file in model path: "
                 << model_path << ", it should contain model_type key.";
    }
    return model_type;
  } else {
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }
}

std::string get_model_backend(const std::filesystem::path& model_path) {
  JsonReader reader;
  // for dit models, the model_index.json file is in the model path
  std::filesystem::path model_index_json_path = model_path / "model_index.json";

  if (std::filesystem::exists(model_index_json_path)) {
    reader.parse(model_index_json_path);

    if (reader.value<std::string>("_diffusers_version").has_value()) {
      return "dit";
    } else {
      LOG(FATAL) << "Please check model_index.json file in model path: "
                 << model_path << ", it should contain _diffusers_version key.";
    }
  }

  // for llm, vlm and rec models, get backend from model type
  std::string model_type = get_model_type(model_path);
  // model_type always exists since get_model_type() will log fatal error if
  // model_type is empty
  return ModelRegistry::get_model_backend(model_type);
}

int run() {
  // check if model path exists
  if (!std::filesystem::exists(FLAGS_model)) {
    LOG(FATAL) << "Model path " << FLAGS_model << " does not exist.";
  }

  std::filesystem::path model_path =
      std::filesystem::path(FLAGS_model).lexically_normal();

  if (FLAGS_model_id.empty()) {
    // use last part of the path as model id
    if (model_path.has_filename()) {
      FLAGS_model_id = std::filesystem::path(FLAGS_model).filename();
    } else {
      FLAGS_model_id =
          std::filesystem::path(FLAGS_model).parent_path().filename();
    }
  }

  if (FLAGS_backend.empty()) {
    FLAGS_backend = get_model_backend(model_path);
  }

  if (FLAGS_host.empty()) {
    // set the host to the local IP when the host is empty
    FLAGS_host = net::get_local_ip_addr();
  }

  bool is_local = false;
  if (FLAGS_host != "" &&
      net::extract_ip(FLAGS_master_node_addr) == FLAGS_host) {
    is_local = true;
  } else {
    is_local = false;
  }

  LOG(INFO) << "set worker role to "
            << (is_local ? "local worker" : "remote worker");

  if (FLAGS_backend == "vlm") {
    FLAGS_enable_prefix_cache = false;
    FLAGS_enable_chunked_prefill = false;
  }

  // if max_tokens_per_chunk_for_prefill is not set, set its value to
  // max_tokens_per_batch
  if (FLAGS_max_tokens_per_chunk_for_prefill < 0) {
    FLAGS_max_tokens_per_chunk_for_prefill = FLAGS_max_tokens_per_batch;
  }

  // set enable_mla by model type
  if (FLAGS_backend != "dit") {
    std::string model_type = get_model_type(model_path);
    if (deepseek_like_model_set.find(model_type) !=
        deepseek_like_model_set.end()) {
      FLAGS_enable_mla = true;
    } else {
      FLAGS_enable_mla = false;
    }
  }

  // Create Master
  Options options;
  options.model_path(FLAGS_model)
      .model_id(FLAGS_model_id)
      .task_type(FLAGS_task)
      .devices(FLAGS_devices)
      .draft_model_path(FLAGS_draft_model)
      .draft_devices(FLAGS_draft_devices)
      .backend(FLAGS_backend)
      .limit_image_per_prompt(FLAGS_limit_image_per_prompt)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .max_tokens_per_batch(FLAGS_max_tokens_per_batch)
      .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
      .max_tokens_per_chunk_for_prefill(FLAGS_max_tokens_per_chunk_for_prefill)
      .num_speculative_tokens(FLAGS_num_speculative_tokens)
      .num_request_handling_threads(FLAGS_num_request_handling_threads)
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
      .device_ip("")
      .transfer_listen_port(FLAGS_transfer_listen_port)
      .nnodes(FLAGS_nnodes)
      .node_rank(FLAGS_node_rank)
      .dp_size(FLAGS_dp_size)
      .ep_size(FLAGS_ep_size)
      .xservice_addr(FLAGS_xservice_addr)
      .instance_name(FLAGS_host + ":" + std::to_string(FLAGS_port))
      .enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_pd_ooc(FLAGS_enable_pd_ooc)
      .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .etcd_addr(FLAGS_etcd_addr)
      .enable_service_routing(FLAGS_enable_service_routing ||
                              FLAGS_enable_disagg_pd)
      .tool_call_parser(FLAGS_tool_call_parser)
      .reasoning_parser(FLAGS_reasoning_parser)
      .priority_strategy(FLAGS_priority_strategy)
      .enable_online_preempt_offline(FLAGS_enable_online_preempt_offline)
      .enable_cache_upload(
          (FLAGS_enable_service_routing || FLAGS_enable_disagg_pd) &&
          FLAGS_enable_prefix_cache && FLAGS_enable_cache_upload)
      .host_blocks_factor(FLAGS_host_blocks_factor)
      .enable_kvcache_store(FLAGS_enable_kvcache_store &&
                            FLAGS_enable_prefix_cache &&
                            (FLAGS_host_blocks_factor > 1.0))
      .prefetch_timeout(FLAGS_prefetch_timeout)
      .prefetch_bacth_size(FLAGS_prefetch_bacth_size)
      .layers_wise_copy_batchs(FLAGS_layers_wise_copy_batchs)
      .store_protocol(FLAGS_store_protocol)
      .store_master_server_address(FLAGS_store_master_server_address)
      .store_metadata_server(FLAGS_store_metadata_server)
      .store_local_hostname(FLAGS_store_local_hostname)
      .enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel)
      .enable_profile_step_time(FLAGS_enable_profile_step_time)
      .enable_profile_token_budget(FLAGS_enable_profile_token_budget)
      .enable_latency_aware_schedule(FLAGS_enable_latency_aware_schedule)
      .profile_max_prompt_length(FLAGS_profile_max_prompt_length)
      .enable_profile_kv_blocks(FLAGS_enable_profile_kv_blocks)
      .disable_ttft_profiling(FLAGS_disable_ttft_profiling)
      .enable_forward_interruption(FLAGS_enable_forward_interruption)
      .max_global_ttft_ms(FLAGS_max_global_ttft_ms)
      .max_global_tpot_ms(FLAGS_max_global_tpot_ms)
      .max_requests_per_batch(FLAGS_max_requests_per_batch)
      .enable_continuous_kvcache(FLAGS_enable_continuous_kvcache)
      .enable_shm(FLAGS_enable_shm)
      .is_local(is_local);

  InstanceName::name()->set_name(options.instance_name().value_or(""));

  // working node
  if (options.node_rank() != 0) {
    auto master = std::make_unique<LLMAssistantMaster>(options);
    master->run();
    return 0;
  } else {
    if (FLAGS_random_seed < 0) {
      FLAGS_random_seed = std::random_device{}() % (1 << 30);
    }
  }

  // master node
  auto master = create_master(FLAGS_backend, options);
  master->run();

  // supported models
  std::vector<std::string> model_names = {FLAGS_model_id};
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
  // Check for --help flag before parsing other flags
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      HelpFormatter::print_help();
      return 0;
    }
  }

  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  google::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging("xllm");

  // Check if model path is provided
  if (FLAGS_model.empty()) {
    HelpFormatter::print_error("--model flag is required");
    return 1;
  }

  return run();
}
