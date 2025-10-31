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

#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <memory>
#include <thread>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "models/model_registry.h"
#include "runtime/dit_master.h"
#include "runtime/llm_engine.h"
#include "runtime/llm_master.h"
#include "runtime/rec_engine.h"
#include "runtime/rec_master.h"
#include "runtime/speculative_engine.h"
#include "runtime/vlm_engine.h"
#include "runtime/vlm_master.h"
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"

#if defined(USE_NPU)
#include <pybind11/pybind11.h>
#endif

namespace xllm {

Master::Master(const Options& options, EngineType type) : options_(options) {
  LOG(INFO) << "Master init options: " << options.to_string();

#if defined(USE_NPU)
  if (options.rank_tablefile().has_value()) {
    FLAGS_rank_tablefile = options.rank_tablefile().value();
  }
  if (options.communication_backend().has_value()) {
    FLAGS_communication_backend = options.communication_backend().value();
  }
  if (options.expert_parallel_degree().has_value()) {
    FLAGS_expert_parallel_degree = options.expert_parallel_degree().value();
  }
  if (options.enable_eplb().has_value()) {
    FLAGS_enable_eplb = options.enable_eplb().value();
  }
  if (options.redundant_experts_num().has_value()) {
    FLAGS_redundant_experts_num = options.redundant_experts_num().value();
  }
  if (options.eplb_update_interval().has_value()) {
    FLAGS_eplb_update_interval = options.eplb_update_interval().value();
  }
  if (options.eplb_update_threshold().has_value()) {
    FLAGS_eplb_update_threshold = options.eplb_update_threshold().value();
  }
#endif
  FLAGS_enable_multi_stream_parallel =
      options.enable_multi_stream_parallel() && (options.nnodes() > 1);

  // construct engine
  const auto devices =
      DeviceNameUtils::parse_devices(options_.devices().value_or("auto"));
  LOG(INFO) << "Creating engine with devices: "
            << DeviceNameUtils::to_string(devices);

  if (options_.enable_disagg_pd()) {
    // Enable service routing in disagg pd mode
    options_.enable_service_routing(true);
    if (options_.instance_role() == InstanceRole::PREFILL) {
      // Disable schedule overlap for prefill instance in disagg pd mode
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for prefill instance "
                      "in disagg pd mode.";
    }
  }

  if (type == EngineType::VLM) {
    options_.enable_schedule_overlap(false);
    LOG(WARNING) << "Force to disable schedule overlap for VLM model, not "
                    "supported yet.";
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options.enable_prefix_cache())
        .task_type(options.task_type())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_cache_upload(options_.enable_cache_upload())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .is_local(options_.is_local());

    auto engine = std::make_unique<VLMEngine>(eng_options);
    engine_ = std::move(engine);
  } else if (type == EngineType::SSM) {
    // create a speculative engine if draft model path is provided
    const auto draft_model_path = options_.draft_model_path().value_or("");
    CHECK(!draft_model_path.empty());
    const auto draft_devices = DeviceNameUtils::parse_devices(
        options_.draft_devices().value_or("auto"));
    LOG(INFO) << "Using draft devices: "
              << DeviceNameUtils::to_string(draft_devices);
    runtime::Options spec_options;
    spec_options.model_path(options_.model_path())
        .draft_model_path(draft_model_path)
        .devices(devices)
        .draft_devices(draft_devices)
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .num_speculative_tokens(options_.num_speculative_tokens())
        .task_type(options.task_type())
        .enable_mla(options.enable_mla())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_cache_upload(options_.enable_cache_upload())
        .enable_offline_inference(options_.enable_offline_inference())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .is_local(options_.is_local());

    if (options_.device_ip().has_value()) {
      spec_options.device_ip(options_.device_ip().value());
    }

    auto spec_engine = std::make_unique<SpeculativeEngine>(spec_options);
    engine_ = std::move(spec_engine);
  } else if (type == EngineType::LLM) {
    if (options_.task_type() == "embed") {
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for embedding model, "
                      "avoiding performance degradation.";
    }
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_cache_upload(options_.enable_cache_upload())
        .host_blocks_factor(options_.host_blocks_factor())
        .enable_kvcache_store(options_.enable_kvcache_store())
        .store_protocol(options_.store_protocol())
        .store_master_server_entry(options_.store_master_server_entry())
        .store_metadata_connstring(options_.store_metadata_connstring())
        .enable_continuous_kvcache(options_.enable_continuous_kvcache())
        .enable_offline_inference(options_.enable_offline_inference())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .is_local(options_.is_local());

    if (options_.device_ip().has_value()) {
      eng_options.device_ip(options_.device_ip().value());
    }
    engine_ = std::make_unique<LLMEngine>(eng_options);
  } else if (type == EngineType::REC) {
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_cache_upload(options_.enable_cache_upload())
        .host_blocks_factor(options_.host_blocks_factor())
        .enable_kvcache_store(options_.enable_kvcache_store())
        .store_protocol(options_.store_protocol())
        .store_master_server_entry(options_.store_master_server_entry())
        .store_metadata_connstring(options_.store_metadata_connstring())
        .enable_continuous_kvcache(options_.enable_continuous_kvcache());
    engine_ = std::make_unique<RecEngine>(eng_options);
  } else {
    LOG(WARNING) << "Not supported llm engine type: "
                 << static_cast<size_t>(type);
  }
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  } else if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  } else if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  } else if (backend == "rec") {
    return std::make_unique<RecMaster>(options);
  } else {
    LOG(FATAL) << "Failed to create master, backend is" << backend;
    return nullptr;
  }
}

}  // namespace xllm
