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
#include <unordered_set>

#include "api_service/api_service.h"
#include "core/common/global_flags.h"
#include "core/common/help_formatter.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/master.h"
#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/options.h"
#include "core/framework/xtensor/xtensor_allocator.h"
#include "core/util/device_name_utils.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "function_call/function_call_parser.h"
#include "parser/reasoning_parser.h"
#include "server/xllm_server_registry.h"
using namespace xllm;

static std::atomic<uint32_t> signal_received{0};

static const std::unordered_set<std::string> prefill_sp_supported_model_set = {
    "deepseek_v32",
    "glm_moe_dsa"};

void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

void validate_flags(const std::string& model_type) {
  if (FLAGS_backend.empty()) {
    LOG(FATAL) << "Model is not supported currently, model type: "
               << model_type;
  }
  if (FLAGS_enable_prefill_sp &&
      !prefill_sp_supported_model_set.contains(model_type)) {
    LOG(FATAL) << "enable_prefill_sp is not supported for model_type="
               << model_type;
  }
#if defined(USE_MLU)
  // Disable enable_schedule_overlap for VLM models on MLU backend
  if (FLAGS_enable_schedule_overlap && FLAGS_backend == "vlm") {
    LOG(WARNING) << "enable_schedule_overlap is not supported for VLM models "
                    "on MLU backend. "
                 << "Disabling enable_schedule_overlap.";
    FLAGS_enable_schedule_overlap = false;
  }
  // TODO: support other block sizes in the future
  if (FLAGS_block_size != 16 && FLAGS_block_size != 1 &&
      FLAGS_backend != "dit") {
    LOG(FATAL) << "Currently, block_size must be 16 for MLU backend, we will "
                  "support other block sizes in the future.";
  }
#endif

#if defined(USE_NPU)
  // enable_xtensor / enable_rolling_load imply enable_manual_loader
  if ((FLAGS_enable_xtensor || FLAGS_enable_rolling_load) &&
      !FLAGS_enable_manual_loader) {
    LOG(WARNING) << "enable_xtensor or enable_rolling_load requires "
                    "enable_manual_loader; forcing enable_manual_loader=true.";
    FLAGS_enable_manual_loader = true;
  }
  if (FLAGS_enable_rolling_load && FLAGS_rolling_load_num_cached_layers < 1) {
    LOG(FATAL) << "rolling_load_num_cached_layers must be >= 1.";
  }
  if (FLAGS_enable_rolling_load && FLAGS_rolling_load_num_rolling_slots < -1) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be >= -1.";
  }
  if (FLAGS_enable_rolling_load && FLAGS_rolling_load_num_rolling_slots >= 0 &&
      FLAGS_rolling_load_num_rolling_slots >
          FLAGS_rolling_load_num_cached_layers) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be <= "
               << "rolling_load_num_cached_layers.";
  }
#else
  if (FLAGS_enable_xtensor) {
    LOG(FATAL) << "enable_xtensor is only supported on NPU.";
  }
  if (FLAGS_enable_manual_loader) {
    LOG(FATAL) << "enable_manual_loader is only supported on NPU.";
  }
  if (FLAGS_enable_rolling_load) {
    LOG(FATAL) << "enable_rolling_load is only supported on NPU.";
  }
#endif
}

int run() {
  // check if model path exists
  if (!std::filesystem::exists(FLAGS_model)) {
    LOG(FATAL) << "Model path " << FLAGS_model << " does not exist.";
  }

  std::filesystem::path model_path =
      std::filesystem::path(FLAGS_model).lexically_normal();
  const std::string default_model_name = xllm::util::get_model_name(model_path);

  if (FLAGS_model_id.empty()) {
    // use last part of the path as model id
    FLAGS_model_id = default_model_name;
  }

  if (FLAGS_backend.empty()) {
    FLAGS_backend = xllm::util::get_model_backend(model_path);
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

// disable block copy kernel on unsupported backends
#if !defined(USE_NPU) && !defined(USE_CUDA)
  FLAGS_enable_block_copy_kernel = false;
#endif
  std::string model_type = "";
  if (FLAGS_backend != "dit") {
    model_type = xllm::util::get_model_type(model_path);
    FLAGS_tool_call_parser = function_call::FunctionCallParser::get_parser_auto(
        FLAGS_tool_call_parser, model_type);
    FLAGS_reasoning_parser =
        ReasoningParser::get_parser_auto(FLAGS_reasoning_parser, model_type);
  }

  // validate flags before creating master
  validate_flags(model_type);

  // Create Master
  Options options;
#if defined(USE_NPU)
  options.npu_kernel_backend(FLAGS_npu_kernel_backend);
#endif
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
      .speculative_algorithm(FLAGS_speculative_algorithm)
      .speculative_suffix_cache_max_depth(
          FLAGS_speculative_suffix_cache_max_depth)
      .speculative_suffix_max_spec_factor(
          FLAGS_speculative_suffix_max_spec_factor)
      .speculative_suffix_max_spec_offset(
          FLAGS_speculative_suffix_max_spec_offset)
      .speculative_suffix_min_token_prob(
          FLAGS_speculative_suffix_min_token_prob)
      .speculative_suffix_max_cached_requests(
          FLAGS_speculative_suffix_max_cached_requests)
      .speculative_suffix_use_tree_spec(FLAGS_speculative_suffix_use_tree_spec)
      .num_request_handling_threads(FLAGS_num_request_handling_threads)
      .communication_backend(FLAGS_communication_backend)
      .enable_eplb(FLAGS_enable_eplb)
      .redundant_experts_num(FLAGS_redundant_experts_num)
      .eplb_update_interval(FLAGS_eplb_update_interval)
      .eplb_update_threshold(FLAGS_eplb_update_threshold)
      .rank_tablefile(FLAGS_rank_tablefile)
      .expert_parallel_degree(FLAGS_expert_parallel_degree)
      .enable_chunked_prefill(FLAGS_enable_chunked_prefill)
      .enable_prefill_sp(FLAGS_enable_prefill_sp)
      .master_node_addr(FLAGS_master_node_addr)
      .instance_role(InstanceRole(FLAGS_instance_role))
      .device_ip("")
      .transfer_listen_port(FLAGS_transfer_listen_port)
      .nnodes(FLAGS_nnodes)
      .node_rank(FLAGS_node_rank)
      .dp_size(FLAGS_dp_size)
      .cp_size(FLAGS_cp_size)
      .ep_size(FLAGS_ep_size)
      .tp_size(FLAGS_tp_size)
      .sp_size(FLAGS_sp_size)
      .cfg_size(FLAGS_cfg_size)
      .instance_name(FLAGS_host + ":" + std::to_string(FLAGS_port))
      .enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_pd_ooc(FLAGS_enable_pd_ooc)
      .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
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
      .enable_graph(FLAGS_enable_graph)
      .max_global_ttft_ms(FLAGS_max_global_ttft_ms)
      .max_global_tpot_ms(FLAGS_max_global_tpot_ms)
      .max_requests_per_batch(FLAGS_max_requests_per_batch)
      .enable_shm(FLAGS_enable_shm)
      .input_shm_size(FLAGS_input_shm_size)
      .output_shm_size(FLAGS_output_shm_size)
      .beam_width(FLAGS_beam_width)
      .kv_cache_dtype(FLAGS_kv_cache_dtype)
      .rec_worker_max_concurrency(FLAGS_rec_worker_max_concurrency)
      .is_local(is_local);

  InstanceName::name()->set_name(options.instance_name().value_or(""));

  // master node
  // init XTensor allocator and PhyPagePool for xtensor mode
  if (FLAGS_enable_xtensor) {
    // Parse devices
    const auto devices =
        DeviceNameUtils::parse_devices(options.devices().value_or("auto"));

    // Initialize XTensorAllocator with first device
    auto& allocator = XTensorAllocator::get_instance();
    allocator.init(devices[0]);

    // Setup distributed XTensor service for multi-GPU/multi-node
    if (FLAGS_nnodes > 1) {
      xtensor::Options xtensor_options;
      xtensor_options.devices(devices)
          .nnodes(FLAGS_nnodes)
          .node_rank(FLAGS_node_rank);
      allocator.setup_multi_node_xtensor_dist(
          xtensor_options, FLAGS_xtensor_master_node_addr, FLAGS_dp_size);
    }

    // Initialize PhyPagePool on all workers
    int64_t num_pages = allocator.init_phy_page_pools(
        FLAGS_max_memory_utilization, FLAGS_max_cache_size);
    if (num_pages <= 0) {
      LOG(FATAL) << "Failed to initialize PhyPagePool";
    }
    LOG(INFO) << "XTensor initialized with " << num_pages << " physical pages";
  }

  std::unique_ptr<Master> master;
  // working node
  if (options.node_rank() != 0) {
    if (FLAGS_backend == "dit") {
      master = std::make_unique<DiTAssistantMaster>(options);
    } else {
      master = std::make_unique<LLMAssistantMaster>(options);
    }
  } else {
    if (FLAGS_random_seed < 0) {
      FLAGS_random_seed = std::random_device{}() % (1 << 30);
    }
    // master node
    master = create_master(FLAGS_backend, options);
  }
  master->run();

  // supported models
  std::vector<std::string> model_names = {FLAGS_model_id};
  std::string model_version = default_model_name;
  std::vector<std::string> model_versions = {model_version};

  if (FLAGS_node_rank == 0 || FLAGS_enable_xtensor) {
    auto api_service =
        std::make_unique<APIService>(master.get(), model_names, model_versions);
    auto xllm_server =
        ServerRegistry::get_instance().register_server("HttpServer");

    // start brpc server
    if (!xllm_server->start(std::move(api_service))) {
      LOG(ERROR) << "Failed to start brpc server on port " << FLAGS_port;
      return -1;
    }
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
