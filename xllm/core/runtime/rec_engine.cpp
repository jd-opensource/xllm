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

#include "rec_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>

#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/parallel_state.h"
#include "util/pretty_print.h"
#include "util/timer.h"
#include "util/utils.h"
#include "worker.h"

namespace xllm {

RecEngine::RecEngine(const runtime::Options& options) : options_(options) {
  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
  }

  // initialize process groups if there are multiple devices
  if (devices.size() > 1) {
    // create a process group for each device if there are multiple gpus
    process_groups_ = parallel_state::create_npu_process_groups(devices);
  }

  WorkerType worker_type = WorkerType::REC;
  const int32_t world_size = static_cast<int32_t>(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg = world_size > 1 ? process_groups_[i].get() : nullptr;
    ParallelArgs parallel_args(rank, world_size, pg);
    workers_.emplace_back(std::make_unique<Worker>(
        parallel_args, devices[i], options_, worker_type));
  }

  if (workers_.size() > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // wait up to 4 seconds for all futures to complete
    folly::collectAll(futures).within(std::chrono::seconds(4)).get();
  }
}

bool RecEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  auto kv_cache_cap = estimate_kv_cache_capacity();

  if (!allocate_kv_cache(kv_cache_cap)) {
    LOG(ERROR) << "Failed to allocate kv cache";
    return false;
  }

  return true;
}

bool RecEngine::init_model() {
  const std::string& model_path = options_.model_path();
  auto model_loader = ModelLoader::create(model_path);
  LOG(INFO) << "Initializing model from: " << model_path;

  // RecEngine does not use tokenizer
  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  // compute the number of local kv heads and head dim
  const int world_size = static_cast<int>(workers_.size());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = xllm::util::parse_dtype(args_.dtype(), options_.devices()[0]);

  // key + value for all layers
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_local_kv_heads: " << n_local_kv_heads_
            << ", head_dim: " << head_dim_ << ", n_layers: " << args_.n_layers()
            << ", dtype: " << dtype_;

  // RecEngine does not use tokenizer, skip vocab_size check

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
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

Engine::KVCacheCapacity RecEngine::estimate_kv_cache_capacity() {
  const int64_t max_cache_size = options_.max_cache_size();
  const double max_memory_utilization = options_.max_memory_utilization();

  const auto& device = workers_[0]->device();
  // call worker to profile memory usage
  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->estimate_kv_cache_capacity_async());
  }

  // pick smallest available memory from all devices
  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    const auto device = workers_[i]->device();
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for device: " << device;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << device
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ", Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
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

  KVCacheCapacity kv_cache_cap;
  kv_cache_cap.cache_size_in_bytes = std::max(cache_size_in_bytes, int64_t(0));
  CHECK_GT(kv_cache_cap.cache_size_in_bytes, 0)
      << "Available kv cache size must be greater than 0";

  // compute kv cache slot size
  const auto dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  // key + value for all layers
  const int64_t slot_size =
      2 * n_local_kv_heads_ * head_dim_ * args_.n_layers() * dtype_size;
  kv_cache_cap.slot_size = slot_size;

  // compute kv blocks num
  const int32_t block_size = options_.block_size();
  const int64_t block_size_in_bytes = block_size * slot_size;
  kv_cache_cap.n_blocks = cache_size_in_bytes / block_size_in_bytes;
  CHECK_GT(kv_cache_cap.n_blocks, 0) << "no n_blocks for kv cache";

  return kv_cache_cap;
}

bool RecEngine::allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap) {
  LOG(INFO) << "kv cache capacity: "
            << "bytes: " << kv_cache_cap.cache_size_in_bytes
            << ", blocks: " << kv_cache_cap.n_blocks
            << ", slot_size: " << kv_cache_cap.slot_size;

  const int32_t block_size = options_.block_size();

  // init kv cache for each worker
  std::vector<std::vector<int64_t>> kv_cache_shape;
  kv_cache_shape.reserve(2);
  kv_cache_shape.emplace_back(std::vector<int64_t>{
      kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
  kv_cache_shape.emplace_back(std::vector<int64_t>{
      kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});

  LOG(INFO) << "Initializing k cache with shape: [" << kv_cache_shape[0] << "]";
  LOG(INFO) << "Initializing v cache with shape: [" << kv_cache_shape[1] << "]";

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks)
      .host_num_blocks(kv_cache_cap.n_blocks)
      .block_size(block_size)
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload());
  kv_cache_manager_ = std::make_unique<BlockManagerPool>(options);

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
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

// RecEngine executes model three times: prefill + 2 decode steps
ForwardOutput RecEngine::step(std::vector<Batch>& batches) {
  if (workers_.empty()) {
    // empty worker, return
    return {};
  }

  Timer timer;
  auto forward_inputs = workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_microseconds, timer.elapsed_microseconds());

  if (!forward_inputs.token_ids.defined()) {
    // empty input, just return
    return {};
  }

  timer.reset();
  // Prefill step: Run the first model execution
  const auto& prefill_output = get_model_output(forward_inputs);
  COUNTER_ADD(rec_first_token_latency_microseconds,
              timer.elapsed_microseconds());

  timer.reset();
  batches[0].process_sample_output(prefill_output.sample_output, false);
  COUNTER_ADD(rec_sampling_latency_microseconds, timer.elapsed_microseconds());

  // Decode steps: Run the model 2 more times for decoding
  ForwardOutput decode_output;

  for (int i = 0; i < 2; ++i) {
    timer.reset();
    forward_inputs = workers_[0]->prepare_inputs(batches[0]);
    COUNTER_ADD(prepare_input_latency_microseconds,
                timer.elapsed_microseconds());

    timer.reset();
    decode_output = get_model_output(forward_inputs);
    if (i == 0) {
      COUNTER_ADD(rec_second_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (i == 1) {
      COUNTER_ADD(rec_third_token_latency_microseconds,
                  timer.elapsed_microseconds());
    }

    timer.reset();
    batches[0].process_sample_output(decode_output.sample_output, false);
    COUNTER_ADD(rec_sampling_latency_microseconds,
                timer.elapsed_microseconds());
  }

  batches[0].finish();

  // Return the final model output
  return decode_output;
}

void RecEngine::update_last_step_result(std::vector<Batch>& batch) {}

std::vector<int64_t> RecEngine::get_active_activation_memory() const {
  // call worker to get active activation memory
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(workers_.size());
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

ForwardOutput RecEngine::get_model_output(const ForwardInput& model_inputs) {
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  futures.reserve(workers_.size());
  // TODO to adapt multi stream parallel later
  BatchedForwardInputs batched_fwd_inputs;
  batched_fwd_inputs.micro_inputs = {model_inputs};
  for (auto& worker : workers_) {
    futures.emplace_back(worker->step_async(batched_fwd_inputs));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the driver
  auto forward_output = results.front().value();

  DCHECK(forward_output.has_value()) << "Failed to execute model";
  return forward_output.value();
}

}  // namespace xllm