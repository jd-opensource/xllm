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

#include "llm_engine.h"

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <memory>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/interruption_bus.h"
#include "common/metrics.h"
#include "framework/block/hierarchy_block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/model_loader.h"
#include "framework/xtensor/multi_layer_xtensor_transfer.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/utils.h"

namespace xllm {

LLMEngine::LLMEngine(const runtime::Options& options,
                     std::shared_ptr<DistManager> dist_manager)
    : options_(options), dist_manager_(dist_manager) {
  InterruptionBus::get_instance().subscribe([this](bool interrupted) {
    this->layer_forward_interrupted_ = interrupted;
  });
  auto master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty())
      << " LLM need to set master node addr, Please set --master_node_addr.";
  const auto& devices = options_.devices();
  // initialize device monitor
  DeviceMonitor::get_instance().initialize(devices);
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
#if defined(USE_NPU)
    FLAGS_enable_atb_comm_multiprocess =
        options.enable_offline_inference() || (options.nnodes() > 1);
#endif
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);

  dp_size_ = options_.dp_size();
  worker_clients_num_ = worker_clients_.size();
  dp_local_tp_size_ = worker_clients_num_ / dp_size_;

  // create ThreadPool for link cluster
  link_threadpool_ = std::make_unique<ThreadPool>(worker_clients_num_);

  process_group_test();

  // init thread pool
  threadpool_ = std::make_unique<ThreadPool>(16);
}

void LLMEngine::process_group_test() {
#if !defined(USE_NPU)
  // In multi-node serving mode, only driver engine
  // create worker_clients_.
  if (worker_clients_num_ > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(worker_clients_num_);
    for (auto& worker : worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // Wait for all futures to complete with a configurable timeout.
    // The timeout can be adjusted via the
    // XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS environment variable (default: 4
    // seconds). This is particularly important in multi-node multi-device
    // scenarios where network latency may require a longer timeout period.
    const int timeout_seconds = util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
#endif
}

bool LLMEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  if (FLAGS_enable_eplb) {
    int32_t num_layers = args_.n_layers() - args_.first_k_dense_replace();
    int32_t num_experts = args_.n_routed_experts();
    eplb_manager_ = std::make_unique<EplbManager>(
        num_layers, worker_clients_num_, num_experts);
  }

  auto kv_cache_cap = estimate_kv_cache_capacity();

  if (!(FLAGS_enable_continuous_kvcache
            ? allocate_continuous_kv_cache(kv_cache_cap)
            : allocate_kv_cache(kv_cache_cap))) {
    LOG(ERROR) << "Failed to initialize  kv cache";
    return false;
  }

  return true;
}

bool LLMEngine::init_model() {
  const std::string& model_path = options_.model_path();
  auto model_loader = ModelLoader::create(model_path);
  LOG(INFO) << "Initializing model from: " << model_path;

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  // compute the number of local kv heads and head dim
  const int world_size = dp_size_ > 1 ? (dp_local_tp_size_)
                                      : static_cast<int>(worker_clients_num_);
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  n_local_q_heads_ = std::max<int64_t>(1, n_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = util::parse_dtype(args_.dtype(), options_.devices()[0]);

  // key + value for all layers
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_local_kv_heads: " << n_local_kv_heads_
            << ", head_dim: " << head_dim_ << ", n_layers: " << args_.n_layers()
            << ", dtype: " << dtype_;

  if (tokenizer_->vocab_size() != args_.vocab_size()) {
    // use tokenizer vocab size if model vocab size is not set
    if (args_.vocab_size() <= 0) {
      LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab "
                      "size: "
                   << tokenizer_->vocab_size();
      args_.vocab_size(tokenizer_->vocab_size());
    } else {
      LOG(WARNING) << "Vocab size mismatch: tokenizer: "
                   << tokenizer_->vocab_size()
                   << ", model: " << args_.vocab_size();
    }
  }

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;
  LOG(INFO) << "Initializing model with random seed: " << FLAGS_random_seed;

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->init_model_async(model_path, FLAGS_random_seed));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  return true;
}

Engine::KVCacheCapacity LLMEngine::estimate_kv_cache_capacity() {
  const int64_t max_cache_size = options_.max_cache_size();
  const double max_memory_utilization = options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to estimate kv cache capacity for worker: " << i;
      continue;
    }

    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    GAUGE_SET(weight_size_in_kilobytes,
              (total_memory - available_memory) / 1024);
    GAUGE_SET(total_memory_size_in_kilobytes, total_memory / 1024);
    // apply memory cap from config if it is set
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }

  Engine::KVCacheCapacity kv_cache_cap;
  kv_cache_cap.cache_size_in_bytes = std::max(cache_size_in_bytes, int64_t(0));
  CHECK_GT(kv_cache_cap.cache_size_in_bytes, 0)
      << "Available kv cache size must be greater than 0";
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            kv_cache_cap.cache_size_in_bytes / 1024);

  for (auto& device : options_.devices()) {
    DeviceMonitor::get_instance().set_total_kv_cache_memory(
        device.index(), kv_cache_cap.cache_size_in_bytes);
    DeviceMonitor::get_instance().set_total_activation_memory(device.index());
  }

  // compute kv cache slot size
  const int64_t dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  int64_t slot_size = 0;
  int64_t index_slot_size = 0;
  if (FLAGS_enable_mla) {
    slot_size = dtype_size * (args_.kv_lora_rank() + args_.qk_rope_head_dim());
  } else {
    slot_size = 2 * dtype_size * head_dim_ * n_local_kv_heads_;
  }
  if (args_.index_n_heads() > 0) {
    int index_n_head = 1;
    index_slot_size = dtype_size * index_n_head * args_.index_head_dim();
  }
  kv_cache_cap.slot_size = slot_size;
  kv_cache_cap.index_slot_size = index_slot_size;
  kv_cache_cap.n_layers = args_.n_layers();

  if (!FLAGS_enable_continuous_kvcache) {
    // compute kv cache n_blocks
    const int32_t block_size = options_.block_size();
    const int64_t block_size_in_bytes =
        block_size * (slot_size + index_slot_size);
    kv_cache_cap.n_blocks = kv_cache_cap.cache_size_in_bytes /
                            (args_.n_layers() * block_size_in_bytes);
    CHECK_GT(kv_cache_cap.n_blocks, 0) << "no n_blocks for kv cache";
  } else {
    int32_t n_pages =
        kv_cache_cap.cache_size_in_bytes / FLAGS_phy_page_granularity_size;
    if (FLAGS_enable_mla) {
      n_pages -= n_pages % (args_.n_layers());
    } else {
      n_pages -= n_pages % (2 * args_.n_layers());
    }
    kv_cache_cap.n_pages = n_pages;
    CHECK_GT(kv_cache_cap.n_pages, 0) << "no n_pages for kv cache";
  }
  return kv_cache_cap;
}

bool LLMEngine::allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap) {
  LOG(INFO) << "kv cache capacity: "
            << readable_size(kv_cache_cap.cache_size_in_bytes)
            << ", blocks: " << kv_cache_cap.n_blocks
            << ", slot_size: " << kv_cache_cap.slot_size
            << ", n_layers: " << kv_cache_cap.n_layers;

  CHECK_GT(kv_cache_cap.n_blocks, 0) << "no memory for kv cache";
  const int32_t block_size = options_.block_size();
  bool enable_lighting_indexer = args_.index_n_heads() > 1;

  // init kv cache for each worker
  std::vector<std::vector<int64_t>> kv_cache_shape;
  kv_cache_shape.reserve(2);
  if (FLAGS_enable_mla) {
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, 1, args_.kv_lora_rank()});
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, 1, args_.qk_rope_head_dim()});
  } else {
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
  }
  if (enable_lighting_indexer) {
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, 1, args_.index_head_dim()});
  }
#if defined(USE_MLU)
  // transpose kv_cache layout for mlu
  // default layout: [n_blocks, block_size, n_head, head_dim]
  // => mlu layout: [n_blocks, n_head, block_size, head_dim]
  for (auto& shape : kv_cache_shape) {
    std::swap(shape[1], shape[2]);
  }
  if (FLAGS_enable_mla) {
    kv_cache_shape[0][3] = args_.kv_lora_rank() + args_.qk_rope_head_dim();
    kv_cache_shape[1] = std::vector<int64_t>{};
  }
#endif

  LOG(INFO) << "Initializing k cache with shape: [" << kv_cache_shape[0] << "]";
  LOG(INFO) << "Initializing v cache with shape: [" << kv_cache_shape[1] << "]";
  if (enable_lighting_indexer) {
    LOG(INFO) << "Initializing indexer cache with shape: [" << kv_cache_shape[2]
              << "]";
  }

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks)
      .block_size(block_size)
      .host_num_blocks(kv_cache_cap.n_blocks * options_.host_blocks_factor())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload())
      .enable_kvcache_store(options_.enable_kvcache_store());
  if (options_.host_blocks_factor() > 1.0 || options_.enable_kvcache_store()) {
    kv_cache_manager_ =
        std::make_unique<HierarchyBlockManagerPool>(options, this, dp_size_);
  } else {
    kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, dp_size_);
  }

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  if (options_.instance_role() == InstanceRole::DEFAULT) {
    for (auto& worker : worker_clients_) {
      futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
    }
  } else {
    if (!options_.device_ip().has_value()) {
      LOG(ERROR)
          << "KVCacheTransfer required device_ip, current value is empty.";
      return false;
    }
    for (auto& worker : worker_clients_) {
      futures.push_back(worker->allocate_kv_cache_with_transfer_async(
          kv_cache_cap.cache_size_in_bytes, kv_cache_shape));
    }
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

bool LLMEngine::allocate_continuous_kv_cache(
    const Engine::KVCacheCapacity& kv_cache_cap) {
  LOG(INFO) << "kv cache capacity: "
            << "bytes: " << kv_cache_cap.cache_size_in_bytes
            << ", blocks: " << kv_cache_cap.n_blocks
            << ", slot_size: " << kv_cache_cap.slot_size;

  std::vector<XTensor::Options> xtensor_options_vec;
  xtensor_options_vec.reserve(2);
  // int64_t head_dim = head_dim_;
  // if (options_.enable_mla()) {
  //   head_dim = args_.kv_lora_rank() + args_.qk_rope_head_dim();
  // }

  XTensor::Options k_xtensor_options;
  XTensor::Options v_xtensor_options;
  k_xtensor_options.num_kv_heads(n_local_kv_heads_)
      .max_context_len(args_.max_position_embeddings())
      .max_seqs_per_batch(options_.max_seqs_per_batch());
  v_xtensor_options.num_kv_heads(n_local_kv_heads_)
      .max_context_len(args_.max_position_embeddings())
      .max_seqs_per_batch(options_.max_seqs_per_batch());
  if (FLAGS_enable_mla) {
    k_xtensor_options.head_size(args_.kv_lora_rank());
    v_xtensor_options.head_size(args_.qk_rope_head_dim());
  } else {
    k_xtensor_options.head_size(head_dim_);
    v_xtensor_options.head_size(head_dim_);
  }

  xtensor_options_vec.emplace_back(k_xtensor_options);
  xtensor_options_vec.emplace_back(v_xtensor_options);

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(
        worker->allocate_continuous_kv_cache_async(xtensor_options_vec));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  int64_t cache_size_per_token = 0;
  if (FLAGS_enable_mla) {
    cache_size_per_token =
        args_.kv_lora_rank() * torch::scalarTypeToTypeMeta(dtype_).itemsize();
  } else {
    cache_size_per_token = kv_cache_cap.slot_size / 2;
  }

  FLAGS_cache_size_per_token = cache_size_per_token;

  // init xtensor manager pool
  xtensor::Options xtensor_manager_options;
  xtensor_manager_options.devices(options_.devices())
      .num_total_pages(kv_cache_cap.n_pages)
      .num_layers(args_.n_layers())
      .cache_size_per_token(cache_size_per_token)
      .server_idx(options_.server_idx());
  kv_cache_manager_ = std::make_unique<XTensorManagerPool>(
      xtensor_manager_options, options_.dp_size());
  return true;
}

bool LLMEngine::pull_kv_blocks(const int32_t src_dp_size,
                               const int32_t src_dp_rank,
                               const std::vector<uint64_t>& src_cluster_ids,
                               const std::vector<std::string>& src_addrs,
                               const std::vector<int64_t>& src_k_cache_ids,
                               const std::vector<int64_t>& src_v_cache_ids,
                               const std::vector<uint64_t>& src_blocks,
                               const int32_t dst_dp_rank,
                               const std::vector<uint64_t>& dst_blocks) {
  int32_t src_world_size = src_cluster_ids.size();
  int32_t src_tp_size = src_world_size / src_dp_size;
  int32_t dst_world_size = options_.nnodes();
  int32_t dst_tp_size = dst_world_size / dp_size_;

  std::vector<bool> results;
  results.reserve(dst_tp_size);
  // Pull the KV cache for all workers in the current DP rank.
  for (size_t tp_rank = 0; tp_rank < dst_tp_size; ++tp_rank) {
    int32_t dst_worker_rank = dst_dp_rank * dst_tp_size + tp_rank;
    // Determine the ranks of the remote workers connected to the current
    // worker.
    int32_t src_dp_worker_rank = dst_worker_rank % src_tp_size;
    int32_t src_worker_rank = src_dp_rank * src_tp_size + src_dp_worker_rank;
    results.push_back(worker_clients_[dst_worker_rank]->pull_kv_blocks(
        src_cluster_ids[src_worker_rank],
        src_addrs[src_worker_rank],
        src_k_cache_ids[src_worker_rank],
        src_v_cache_ids[src_worker_rank],
        src_blocks,
        dst_blocks));
  }

  for (bool result : results) {
    if (!result) {
      return false;
    }
  }
  return true;
}

std::vector<folly::SemiFuture<uint32_t>> LLMEngine::transfer_kv_blocks(
    const uint32_t dp_rank,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::vector<folly::SemiFuture<uint32_t>> futures;
  futures.reserve(dp_local_tp_size_);

  for (auto tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    futures.emplace_back(worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]
                             ->transfer_kv_blocks(block_transfer_info));
  }

  return std::move(futures);
}

void LLMEngine::transfer_kv_blocks(
    const uint32_t dp_rank,
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  for (auto tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]->transfer_kv_blocks(
        batch_id, block_transfer_info);
  }
}

void LLMEngine::prefetch_from_storage(
    const uint32_t dp_rank,
    const std::vector<BlockTransferInfo>& block_transfer_info,
    std::shared_ptr<std::atomic<int32_t>> flag,
    std::vector<std::shared_ptr<std::atomic<uint32_t>>>* prefetch_results) {
  prefetch_results->reserve(dp_local_tp_size_);
  flag->store(dp_local_tp_size_, std::memory_order_relaxed);
  for (auto tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    prefetch_results->emplace_back(std::make_shared<std::atomic<uint32_t>>(0));
    worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]
        ->prefetch_from_storage(
            block_transfer_info, flag, prefetch_results->at(tp_rank));
  }
}

void LLMEngine::get_device_info(std::vector<std::string>& device_ips,
                                std::vector<uint16_t>& ports) {
  if (worker_device_ips_.size() != worker_clients_num_ ||
      worker_ports_.size() != worker_clients_num_) {
    worker_device_ips_.reserve(worker_clients_num_);
    worker_ports_.reserve(worker_clients_num_);
    for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
         ++worker_rank) {
      std::string device_ip;
      uint16_t port;
      worker_clients_[worker_rank]->get_device_info(device_ip, port);
      worker_device_ips_.emplace_back(std::move(device_ip));
      worker_ports_.emplace_back(port);
    }
  }

  device_ips = worker_device_ips_;
  ports = worker_ports_;
}

void LLMEngine::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<int64_t>& k_cache_ids,
                               std::vector<int64_t>& v_cache_ids) {
  cluster_ids.reserve(worker_clients_num_);
  addrs.reserve(worker_clients_num_);
  k_cache_ids.reserve(worker_clients_num_);
  v_cache_ids.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    uint64_t cluster_id;
    std::string addr;
    int64_t k_cache_id;
    int64_t v_cache_id;
    worker_clients_[worker_rank]->get_cache_info(
        cluster_id, addr, k_cache_id, v_cache_id);
    cluster_ids.emplace_back(cluster_id);
    addrs.emplace_back(addr);
    k_cache_ids.emplace_back(k_cache_id);
    v_cache_ids.emplace_back(v_cache_id);
  }
}

bool LLMEngine::link_cluster(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& addrs,
                             const std::vector<std::string>& device_ips,
                             const std::vector<uint16_t>& ports,
                             const int32_t src_dp_size) {
  // Indicate which worker in the dp group in prefill the current worker needs
  // to connect to. First, we connect the rank 0 workers in each DP. Then,
  // increment the ranks sequentially.
  int32_t src_dp_worker_index = 0;
  int32_t src_world_size = cluster_ids.size();
  int32_t src_tp_size = src_world_size / src_dp_size;

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    // The worker for decoding needs to establish a connection for each dp group
    // in prefill.
    std::vector<uint64_t> dp_cluster_ids;
    std::vector<std::string> dp_addrs;
    std::vector<std::string> dp_device_ips;
    std::vector<uint16_t> dp_ports;
    dp_cluster_ids.reserve(src_dp_size);
    dp_addrs.reserve(src_dp_size);
    dp_device_ips.reserve(src_dp_size);
    dp_ports.reserve(src_dp_size);
    for (int32_t i = 0; i < src_dp_size; ++i) {
      int32_t src_worker_index = i * src_tp_size + src_dp_worker_index;
      dp_cluster_ids.emplace_back(cluster_ids[src_worker_index]);
      dp_addrs.emplace_back(addrs[src_worker_index]);
      dp_device_ips.emplace_back(device_ips[src_worker_index]);
      dp_ports.emplace_back(ports[src_worker_index]);
    }
    // Increment the rank.
    src_dp_worker_index = (src_dp_worker_index + 1) % src_tp_size;

    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                dp_cluster_ids = std::move(dp_cluster_ids),
                                dp_addrs = std::move(dp_addrs),
                                dp_device_ips = std::move(dp_device_ips),
                                dp_ports = std::move(dp_ports)]() mutable {
      promise.setValue(worker_clients_[worker_rank]->link_cluster(
          dp_cluster_ids, dp_addrs, dp_device_ips, dp_ports));
    });
    futures.emplace_back(std::move(future));
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Link cluster failed.";
      return false;
    }
  }
  return true;
}

bool LLMEngine::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<std::string>& device_ips,
                               const std::vector<uint16_t>& ports,
                               const int32_t src_dp_size) {
  // Indicate which worker in the dp group in prefill the current worker needs
  // to unlink.
  int32_t src_dp_worker_index = 0;
  int32_t src_world_size = cluster_ids.size();
  int32_t src_tp_size = src_world_size / src_dp_size;

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    // The worker for decoding needs to unlink for each dp group in prefill.
    std::vector<uint64_t> dp_cluster_ids;
    std::vector<std::string> dp_addrs;
    std::vector<std::string> dp_device_ips;
    std::vector<uint16_t> dp_ports;
    dp_cluster_ids.reserve(src_dp_size);
    dp_addrs.reserve(src_dp_size);
    dp_device_ips.reserve(src_dp_size);
    dp_ports.reserve(src_dp_size);
    for (int32_t i = 0; i < src_dp_size; ++i) {
      int32_t src_worker_index = i * src_tp_size + src_dp_worker_index;
      dp_cluster_ids.emplace_back(cluster_ids[src_worker_index]);
      dp_addrs.emplace_back(addrs[src_worker_index]);
      dp_device_ips.emplace_back(device_ips[src_worker_index]);
      dp_ports.emplace_back(ports[src_worker_index]);
    }
    src_dp_worker_index = (src_dp_worker_index + 1) % src_tp_size;

    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                dp_cluster_ids = std::move(dp_cluster_ids),
                                dp_addrs = std::move(dp_addrs),
                                dp_device_ips = std::move(dp_device_ips),
                                dp_ports = std::move(dp_ports)]() mutable {
      promise.setValue(worker_clients_[worker_rank]->unlink_cluster(
          dp_cluster_ids, dp_addrs, dp_device_ips, dp_ports));
    });
    futures.emplace_back(std::move(future));
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Unlink cluster failed.";
      return false;
    }
  }
  return true;
}

ForwardOutput LLMEngine::step(std::vector<Batch>& batch) {
  if (worker_clients_.empty()) {
    // empty worker, return
    return {};
  }
  Timer timer;
  DCHECK(dp_size_ == batch.size())
      << "Split DP batch failed with dp_size as " << dp_size_
      << " and actual batch size as " << batch.size() << ".";

  auto raw_forward_inputs = prepare_inputs(batch);
  DCHECK(dp_size_ == raw_forward_inputs.size())
      << "The processed raw forward inputs size " << raw_forward_inputs.size()
      << " is not equal to dp size " << dp_size_ << ".";

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);

  // update dp related global paramters and then execute model
  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    auto dp_rank = worker_rank / dp_local_tp_size_;
    futures.emplace_back(
        worker_clients_[worker_rank]->step_async(raw_forward_inputs[dp_rank]));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  if (FLAGS_enable_eplb && !options_.enable_schedule_overlap()) {
    process_eplb_data(results);
  }

  assert(dp_size_ == worker_clients_num_ / dp_local_tp_size_);
  size_t dp_rank = 0;
  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_tp_size_) {
    auto result = results[worker_rank].value();
    if (result.has_value()) {
      if (result.value().outputs.empty() && layer_forward_interrupted_) {
        throw ForwardInterruptedException();
      }
      // if src_seq_idxes is not empty, skip sample output processing and
      // process beam search output instead
      if (result.value().src_seq_idxes.size() == 0) {
        // set second input param enable_schedule_overlap to false,
        // if it's not enabled, process_sample_output will append the real
        // token, if it's enabled, this false here will append the fake token in
        // process_sample_output
        batch[dp_rank].process_sample_output(result.value(), false);
      } else {
        batch[dp_rank].process_beam_search_output(result.value(), false);
      }
    } else {
      LOG(FATAL) << "Failed to execute model, result has no value";
    }
    ++dp_rank;
  }

  COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());
  return {};
}

void LLMEngine::update_last_step_result(std::vector<Batch>& last_batch) {
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);
  std::vector<RawForwardOutput> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size_);

  // NOTE: We only need to get the output from the driver worker,
  // cause the output on other workers is the same as that on driver.
  // Under data parallelism (DP), we need to get dp_size outputs.
  // The `stride` means the workers num we can skip.
  int stride = dp_local_tp_size_;
  // If EPLB is enabled, we need to get results from all workers,
  // because the experts on each worker are different,
  // and the tokens load of all experts needs to be returned to engine.
  // so we can not skip any worker.
  if (FLAGS_enable_eplb) {
    stride = 1;
  }

  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += stride) {
    futures.emplace_back(
        worker_clients_[worker_rank]->get_last_step_result_async());
  }
  // wait for the all future to complete
  auto last_step_results = folly::collectAll(futures).get();

  if (FLAGS_enable_eplb) {
    process_eplb_data(last_step_results);
  }

  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_tp_size_) {
    auto result = last_step_results[worker_rank / stride].value();
    if (result.has_value()) {
      raw_forward_outputs.emplace_back(std::move(result.value()));
    } else {
      LOG(FATAL) << "Failed to get last step results, result has no value";
    }
  }

  for (auto i = 0; i < last_batch.size(); i++) {
    last_batch[i].process_sample_output(raw_forward_outputs[i],
                                        options_.enable_schedule_overlap());
  }
}

std::vector<int64_t> LLMEngine::get_active_activation_memory() const {
  // call worker to get active activation memory
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(worker_clients_num_);
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

void LLMEngine::setup_workers(const runtime::Options& options) {
  if (!dist_manager_) {
    dist_manager_ = std::make_shared<DistManager>(options);
  }
  worker_clients_ = dist_manager_->get_worker_clients();
}

void LLMEngine::process_eplb_data(
    const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results) {
  int32_t num_layers = args_.n_layers() - args_.first_k_dense_replace();
  int32_t num_device_experts = args_.n_routed_experts() / worker_clients_num_ +
                               FLAGS_redundant_experts_num;
  std::vector<torch::Tensor> tensors;
  std::vector<int32_t> layer_ids(results.size(), -1);
  tensors.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < results.size(); ++worker_rank) {
    auto result = results[worker_rank].value();
    if (result.has_value()) {
      tensors.emplace_back(
          torch::from_blob(result.value().expert_load_data.data(),
                           {num_layers, num_device_experts},
                           torch::TensorOptions().dtype(torch::kInt64))
              .clone());
      layer_ids[worker_rank] = result.value().prepared_layer_id;
    } else {
      LOG(ERROR) << "Failed to process EPLB data";
    }
  }
  eplb_manager_->set_prepared_layer_ids(layer_ids);
  eplb_manager_->update_expert_load(tensors);
}

std::vector<RawForwardInput> LLMEngine::prepare_inputs(
    std::vector<Batch>& batch) {
  std::vector<RawForwardInput> batched_inputs;
  batched_inputs.reserve(dp_size_);
  // some dp related variables
  std::vector<int32_t> dp_global_token_nums(dp_size_);
  std::vector<int32_t> dp_is_decode(dp_size_, 0);
  bool global_empty_kv_cache = true;
  // when enable dp, we need to check the forward type of each batch
  // and set the empty forward type of each batch to the same value as the first
  // batch
  BatchForwardType batch_forward_type;

  // build model input for every single micro batch
  for (auto dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    batched_inputs.emplace_back(std::move(
        batch[dp_rank].prepare_forward_input(args_, threadpool_.get())));
    dp_global_token_nums[dp_rank] =
        batched_inputs[dp_rank].flatten_tokens_vec.size();
    global_empty_kv_cache =
        batched_inputs[dp_rank].empty_kv_cache && global_empty_kv_cache;
    if (batch_forward_type.is_empty() &&
        !batched_inputs[dp_rank].batch_forward_type.is_empty()) {
      batch_forward_type = batched_inputs[dp_rank].batch_forward_type;
    }
    dp_is_decode[dp_rank] = batch_forward_type.is_decode() &&
                            batched_inputs[dp_rank].q_max_seq_len == 1;
  }

  // eplb related
  EplbInfo eplb_info;
  if (FLAGS_enable_eplb) {
    eplb_info = eplb_manager_->get_eplb_info();
  }

  // update dp_global_token_nums and global_empty_kv_cache
  for (auto dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    batched_inputs[dp_rank].dp_global_token_nums = dp_global_token_nums;
    batched_inputs[dp_rank].dp_is_decode = dp_is_decode;
    batched_inputs[dp_rank].global_empty_kv_cache = global_empty_kv_cache;
    if (FLAGS_enable_eplb) {
      batched_inputs[dp_rank].eplb_info = eplb_info;
    }
    if (batched_inputs[dp_rank].batch_forward_type.is_empty()) {
      batched_inputs[dp_rank].batch_forward_type = batch_forward_type;
    }
  }

  return batched_inputs;
}

}  // namespace xllm
