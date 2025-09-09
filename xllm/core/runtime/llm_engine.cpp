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
#if defined(USE_NPU)
#include <hccl/hccl.h>
#endif
#include <sys/sysinfo.h>
#include <unistd.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <memory>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/model_loader.h"
#include "framework/page/multi_layer_xtensor_transfer.h"
#include "framework/parallel_state.h"
#include "llm_worker_impl.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/pretty_print.h"
#include "util/utils.h"

namespace xllm {

LLMEngine::LLMEngine(const runtime::Options& options,
                     std::shared_ptr<DistManager> dist_manager)
    : options_(options), dist_manager_(dist_manager) {
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
    FLAGS_enable_atb_comm_multiprocess = (options.nnodes() > 1);
#endif
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);

  // In multi-node serving mode, only driver engine
  // create worker_clients_.
  if (worker_clients_.size() > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(worker_clients_.size());
    for (auto& worker : worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // wait up to 4 seconds for all futures to complete
    folly::collectAll(futures).within(std::chrono::seconds(4)).get();
  }
}

bool LLMEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
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
  const int world_size = options_.dp_size() > 1
                             ? (worker_clients_.size() / options_.dp_size())
                             : static_cast<int>(worker_clients_.size());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  n_local_q_heads_ = std::max<int64_t>(1, n_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = util::parse_dtype(args_.dtype(), options_.devices()[0]);
  if (FLAGS_enable_eplb) {
    int32_t num_layers = args_.n_layers() - args_.first_k_dense_replace();
    int32_t num_experts = args_.n_routed_experts();
    eplb_manager_ = std::make_unique<EplbManager>(
        num_layers, worker_clients_.size(), num_experts);
  }

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

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->init_model_async(model_path));
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
  futures.reserve(worker_clients_.size());
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
  if (FLAGS_enable_mla) {
    slot_size = dtype_size * (args_.kv_lora_rank() + args_.qk_rope_head_dim());
  } else {
    slot_size = 2 * dtype_size * head_dim_ * n_local_kv_heads_;
  }
  kv_cache_cap.slot_size = slot_size;

  if (!FLAGS_enable_continuous_kvcache) {
    // compute kv cache n_blocks
    const int32_t block_size = options_.block_size();
    const int64_t block_size_in_bytes = block_size * slot_size;
    kv_cache_cap.n_blocks = kv_cache_cap.cache_size_in_bytes /
                            (args_.n_layers() * block_size_in_bytes);
    CHECK_GT(kv_cache_cap.n_blocks, 0) << "no n_blocks for kv cache";
  } else {
    int32_t n_pages = kv_cache_cap.cache_size_in_bytes / FLAGS_granularity_size;
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
            << "bytes: " << kv_cache_cap.cache_size_in_bytes
            << ", blocks: " << kv_cache_cap.n_blocks
            << ", slot_size: " << kv_cache_cap.slot_size;

  CHECK_GT(kv_cache_cap.n_blocks, 0) << "no memory for kv cache";
  const int32_t block_size = options_.block_size();

  // init kv cache for each worker
  std::vector<std::vector<int64_t>> kv_cache_shape;
  kv_cache_shape.reserve(2);
  if (FLAGS_enable_mla) {
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, 1, args_.kv_lora_rank()});
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, 1, args_.qk_rope_head_dim()});
  } else {
#if defined(USE_NPU)
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
#elif defined(USE_MLU)
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, n_local_kv_heads_, block_size, head_dim_});
    kv_cache_shape.emplace_back(std::vector<int64_t>{
        kv_cache_cap.n_blocks, n_local_kv_heads_, block_size, head_dim_});
#endif
  }

  LOG(INFO) << "Initializing k cache with shape: [" << kv_cache_shape[0] << "]";
  LOG(INFO) << "Initializing v cache with shape: [" << kv_cache_shape[1] << "]";

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks)
      .block_size(block_size)
      .host_num_blocks(kv_cache_cap.n_blocks * options_.host_blocks_factor())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload())
      .enable_kvcache_store(options_.enable_kvcache_store());
  block_manager_pool_ =
      std::make_unique<BlockManagerPool>(options, options_.dp_size());

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
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

  // init page manager pool
  page::Options page_options;
  page_options.devices(options_.devices())
      .num_total_pages(kv_cache_cap.n_pages)
      .num_layers(args_.n_layers())
      .cache_size_per_token(cache_size_per_token);
  page_manager_pool_ =
      std::make_unique<PageManagerPool>(page_options, options_.dp_size());

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
  int32_t dst_tp_size = dst_world_size / options_.dp_size();

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

std::vector<folly::SemiFuture<uint32_t>>
LLMEngine::load_kv_blocks_from_store_async(
    const uint32_t dp_rank,
    const std::vector<CacheBlockInfo>& cache_block_info) {
  std::vector<folly::SemiFuture<uint32_t>> futures;
  auto tp_size = this->worker_clients_.size() / this->options_.dp_size();

  futures.reserve(tp_size);
  for (auto tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
    futures.emplace_back(
        this->worker_clients_[tp_rank + tp_size * dp_rank]
            ->load_kv_blocks_from_store_async(cache_block_info));
  }
  return std::move(futures);
}

void LLMEngine::get_device_info(std::vector<std::string>& device_ips,
                                std::vector<uint16_t>& ports) {
  device_ips.reserve(worker_clients_.size());
  ports.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
       ++worker_rank) {
    std::string device_ip;
    uint16_t port;
    worker_clients_[worker_rank]->get_device_info(device_ip, port);
    device_ips.emplace_back(std::move(device_ip));
    ports.emplace_back(port);
  }
}

void LLMEngine::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<int64_t>& k_cache_ids,
                               std::vector<int64_t>& v_cache_ids) {
  cluster_ids.reserve(worker_clients_.size());
  addrs.reserve(worker_clients_.size());
  k_cache_ids.reserve(worker_clients_.size());
  v_cache_ids.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
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
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
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

    bool ret = worker_clients_[worker_rank]->link_cluster(
        dp_cluster_ids, dp_addrs, dp_device_ips, dp_ports);
    if (!ret) {
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
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
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

    bool ret = worker_clients_[worker_rank]->unlink_cluster(
        dp_cluster_ids, dp_addrs, dp_device_ips, dp_ports);
    if (!ret) {
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
  DCHECK(options_.dp_size() == batch.size())
      << "Split Dp batch failed with dp_size as " << options_.dp_size()
      << "micro batch size as " << batch.size() << ".";

  auto worker_clients_num = worker_clients_.size();
  auto dp_size = options_.dp_size();
  auto dp_local_tp_size = worker_clients_num / dp_size;
  std::vector<RawForwardInput> raw_forward_inputs;
  raw_forward_inputs.reserve(dp_size);
  std::vector<int32_t> dp_global_token_nums(dp_size);
  bool global_empty_kv_cache = true;
  EplbInfo eplb_info;
  for (auto dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
    // assume the order in workers_ is its rank
    RawForwardInput raw_forward_input = batch[dp_rank].prepare_forward_input();
    raw_forward_inputs.push_back(raw_forward_input);
    dp_global_token_nums[dp_rank] = raw_forward_input.flatten_tokens_vec.size();
    global_empty_kv_cache =
        raw_forward_inputs[dp_rank].empty_kv_cache && global_empty_kv_cache;
  }

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num);
  if (FLAGS_enable_eplb) {
    eplb_info = eplb_manager_->get_eplb_info();
  }
  // update dp related global paramters and then execute model
  for (auto worker_rank = 0; worker_rank < worker_clients_num; ++worker_rank) {
    auto dp_rank = worker_rank / dp_local_tp_size;
    raw_forward_inputs[dp_rank].dp_global_token_nums = dp_global_token_nums;
    raw_forward_inputs[dp_rank].global_empty_kv_cache = global_empty_kv_cache;
    if (FLAGS_enable_eplb) {
      raw_forward_inputs[dp_rank].eplb_info = eplb_info;
    }
    futures.emplace_back(
        worker_clients_[worker_rank]->step_async(raw_forward_inputs[dp_rank]));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  if (FLAGS_enable_eplb && !options_.enable_schedule_overlap()) {
    process_eplb_data(results, worker_clients_num);
  }
  // concat results from dp ranks
  std::vector<std::optional<RawForwardOutput>> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size);
  for (auto worker_rank = 0; worker_rank < worker_clients_num;
       worker_rank += dp_local_tp_size) {
    auto result = results[worker_rank].value();
    if (result.has_value()) {
      raw_forward_outputs.push_back(result);
    } else {
      throw std::runtime_error("Failed to execute model");
    }
  }

  // set second input param enable_schedule_overlap to false,
  // if it's not enabled, process_sample_output will append the real token,
  // if it's enabled, this false here will append the fake token in
  // process_sample_output
  for (auto dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
    batch[dp_rank].process_sample_output(raw_forward_outputs[dp_rank].value(),
                                         false);
  }
  COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());
  return {};
}

void LLMEngine::update_last_step_result(std::vector<Batch>& last_batch) {
  auto dp_size = options_.dp_size();
  auto worker_clients_num = worker_clients_.size();
  auto dp_local_tp_size = worker_clients_num / dp_size;

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(dp_size);
  std::vector<RawForwardOutput> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size);
  if (FLAGS_enable_eplb) {
    for (auto worker_rank = 0; worker_rank < worker_clients_num;
         worker_rank++) {
      futures.emplace_back(
          worker_clients_[worker_rank]->get_last_step_result_async());
    }
    // wait for the all future to complete
    auto last_step_results = folly::collectAll(futures).get();
    // concat last step results from dp ranks
    process_eplb_data(last_step_results, worker_clients_num);
    for (auto worker_rank = 0; worker_rank < worker_clients_num;
         worker_rank += dp_local_tp_size) {
      auto result = last_step_results[worker_rank].value();
      if (result.has_value()) {
        raw_forward_outputs.emplace_back(std::move(result.value()));
      } else {
        throw std::runtime_error("Failed to get last step results.");
      }
    }
  } else {
    for (auto worker_rank = 0; worker_rank < worker_clients_num;
         worker_rank += dp_local_tp_size) {
      futures.emplace_back(
          worker_clients_[worker_rank]->get_last_step_result_async());
    }

    // wait for the all future to complete
    auto last_step_results = folly::collectAll(futures).get();
    // concat last step results from dp ranks

    for (auto worker_rank = 0; worker_rank < worker_clients_num;
         worker_rank += dp_local_tp_size) {
      auto result = last_step_results[worker_rank / dp_local_tp_size].value();
      if (result.has_value()) {
        raw_forward_outputs.emplace_back(std::move(result.value()));
      } else {
        throw std::runtime_error("Failed to get last step results.");
      }
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
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(worker_clients_.size());
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
    const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results,
    int32_t worker_clients_num) {
  int32_t num_layers = args_.n_layers() - args_.first_k_dense_replace();
  int32_t num_device_experts =
      args_.n_routed_experts() / worker_clients_.size() +
      FLAGS_redundant_experts_num;
  std::vector<torch::Tensor> tensors;
  std::vector<int32_t> layer_ids(results.size(), -1);
  tensors.reserve(worker_clients_.size());
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

}  // namespace xllm
