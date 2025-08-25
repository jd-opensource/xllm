#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

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
#include "runtime/llm_engine.h"
#include "runtime/llm_master.h"
#include "runtime/speculative_engine.h"
#include "runtime/vlm_engine.h"
#include "runtime/vlm_master.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {

Master::Master(const Options& options, EngineType type) : options_(options) {
  LOG(INFO) << "Master init options: " << options.to_string();

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
  if (options.eplb_update_rate().has_value()) {
    FLAGS_eplb_update_rate = options.eplb_update_rate().value();
  }
  if (options.eplb_update_threshold().has_value()) {
    FLAGS_eplb_update_threshold = options.eplb_update_threshold().value();
  }

  // construct engine
  const auto devices =
      DeviceNameUtils::parse_devices(options_.devices().value_or("auto"));
  CHECK_GT(devices.size(), 0) << "At least one device is required";
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
        .enable_service_routing(options_.enable_service_routing());

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
        .enable_schedule_overlap(options_.enable_schedule_overlap());
    if (options_.device_ip().has_value()) {
      spec_options.device_ip(options_.device_ip().value());
    }

    auto spec_engine = std::make_unique<SpeculativeEngine>(spec_options);
    engine_ = std::move(spec_engine);
  } else if (type == EngineType::LLM) {
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
        .enable_schedule_overlap(options_.enable_schedule_overlap());
    if (options_.device_ip().has_value()) {
      eng_options.device_ip(options_.device_ip().value());
    }
    engine_ = std::make_unique<LLMEngine>(eng_options);
  } else {
    LOG(FATAL) << "Not supported llm engine type: "
               << static_cast<size_t>(type);
  }
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  } else if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  } else {
    LOG(FATAL) << "Failed to create master, backend is" << backend;
    return nullptr;
  }
}

}  // namespace xllm
