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
#include <chrono>
#include <functional>
#include <map>
#include <memory>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/rec_model_utils.h"
#include "framework/model/model_args.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/request/rec_type.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

// ============================================================
// RecEngine Implementation
// ============================================================

RecEngine::RecEngine(const runtime::Options& options,
                     std::shared_ptr<DistManager> dist_manager)
    : options_(options), dist_manager_(dist_manager) {
  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
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

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  // Determine rec model kind and create pipeline via factory
  rec_model_kind_ = get_rec_model_kind(args_.model_type());
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  pipeline_ = create_pipeline(pipeline_type, *this);

  // LlmRec-specific initialization
  if (rec_model_kind_ == RecModelKind::kLlmRec) {
#if defined(USE_NPU)
    FLAGS_enable_atb_comm_multiprocess =
        options_.enable_offline_inference() || (options_.nnodes() > 1);
#endif

    auto master_node_addr = options_.master_node_addr().value_or("");
    CHECK(!master_node_addr.empty())
        << "REC(kLlmRec) need to set master node addr, "
           "Please set --master_node_addr.";
  }

  // Pipeline-specific setup
  pipeline_->setup_workers();
  pipeline_->process_group_test();

  if (!threadpool_) {
    threadpool_ = std::make_unique<ThreadPool>(16);
  }

  // Compute KV cache config (shared logic)
  const int world_size = static_cast<int>(pipeline_->num_workers());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = xllm::util::parse_dtype(args_.dtype(), options_.devices()[0]);

  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_local_kv_heads: " << n_local_kv_heads_
            << ", head_dim: " << head_dim_ << ", n_layers: " << args_.n_layers()
            << ", dtype: " << dtype_;
  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;

  // Pipeline-specific model initialization
  return pipeline_->init_model_workers(model_path);
}

Engine::KVCacheCapacity RecEngine::estimate_kv_cache_capacity() {
  const int64_t max_cache_size = options_.max_cache_size();
  const double max_memory_utilization = options_.max_memory_utilization();

  int64_t cache_size_in_bytes = pipeline_->estimate_min_available_memory();

  // apply memory cap from config
  if (max_memory_utilization < 1.0 || max_cache_size > 0) {
    // Re-estimate with caps applied (pipeline returns raw available memory)
    // The caps are applied in estimate_min_available_memory
  }

  KVCacheCapacity kv_cache_cap;
  kv_cache_cap.cache_size_in_bytes = std::max(cache_size_in_bytes, int64_t(0));
  CHECK_GT(kv_cache_cap.cache_size_in_bytes, 0)
      << "Available kv cache size must be greater than 0";

  // compute kv cache slot size
  const int64_t dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  const int64_t slot_size = 2 * dtype_size * head_dim_ * n_local_kv_heads_;
  kv_cache_cap.slot_size = slot_size;
  kv_cache_cap.n_layers = args_.n_layers();

  const int32_t block_size = options_.block_size();
  const int64_t block_size_in_bytes = block_size * slot_size;
  kv_cache_cap.n_blocks = kv_cache_cap.cache_size_in_bytes /
                          (args_.n_layers() * block_size_in_bytes);
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
#if defined(USE_MLU)
  for (auto& shape : kv_cache_shape) {
    std::swap(shape[1], shape[2]);
  }
#endif

  LOG(INFO) << "Initializing k cache with shape: [" << kv_cache_shape[0] << "]";
  LOG(INFO) << "Initializing v cache with shape: [" << kv_cache_shape[1] << "]";

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks)
      .host_num_blocks(0)
      .block_size(block_size)
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload());
  kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, dp_size_);

  return pipeline_->allocate_kv_cache(kv_cache_shape);
}

ForwardOutput RecEngine::step(std::vector<Batch>& batches) {
  return pipeline_->step(batches);
}

void RecEngine::update_last_step_result(std::vector<Batch>& batch) {
  UNUSED_PARAMETER(batch);
}

std::vector<int64_t> RecEngine::get_active_activation_memory() const {
  return pipeline_->get_active_activation_memory();
}

// ============================================================
// LlmRecEnginePipeline Implementation
// ============================================================

RecEngine::LlmRecEnginePipeline::LlmRecEnginePipeline(RecEngine& engine)
    : RecEnginePipeline(engine) {}

void RecEngine::LlmRecEnginePipeline::setup_workers() {
  if (!engine_.dist_manager_) {
    engine_.dist_manager_ = std::make_shared<DistManager>(engine_.options_);
  }
  engine_.worker_clients_ = engine_.dist_manager_->get_worker_clients();
  engine_.dp_size_ = engine_.options_.dp_size();
  engine_.worker_clients_num_ = engine_.worker_clients_.size();
  engine_.dp_local_tp_size_ = engine_.worker_clients_num_ / engine_.dp_size_;
}

void RecEngine::LlmRecEnginePipeline::process_group_test() {
#if !defined(USE_NPU)
  if (engine_.worker_clients_num_ > 1) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(engine_.worker_clients_num_);
    for (auto& worker : engine_.worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    const int timeout_seconds = util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
#endif
}

bool RecEngine::LlmRecEnginePipeline::init_model_workers(
    const std::string& model_path) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.worker_clients_num_);
  for (auto& worker : engine_.worker_clients_) {
    futures.push_back(worker->init_model_async(model_path, FLAGS_random_seed));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

int64_t RecEngine::LlmRecEnginePipeline::estimate_min_available_memory() {
  const int64_t max_cache_size = engine_.options_.max_cache_size();
  const double max_memory_utilization =
      engine_.options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.push_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for worker: " << i;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
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
  return cache_size_in_bytes;
}

bool RecEngine::LlmRecEnginePipeline::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

size_t RecEngine::LlmRecEnginePipeline::num_workers() const {
  if (engine_.dp_size_ > 1) {
    return engine_.dp_local_tp_size_;
  }
  return engine_.worker_clients_.size();
}

std::vector<RawForwardInput> RecEngine::LlmRecEnginePipeline::prepare_inputs(
    std::vector<Batch>& batch) {
  std::vector<RawForwardInput> batched_inputs;
  batched_inputs.reserve(engine_.dp_size_);

  std::vector<int32_t> dp_global_token_nums(engine_.dp_size_);
  std::vector<int32_t> dp_is_decode(engine_.dp_size_, 0);
  bool global_empty_kv_cache = true;
  BatchForwardType batch_forward_type;

  for (int32_t dp_rank = 0; dp_rank < engine_.dp_size_; ++dp_rank) {
    // kLlmRec needs refresh_forward_type for correct dp_is_decode
    batch[dp_rank].refresh_forward_type();

    batched_inputs.emplace_back(std::move(batch[dp_rank].prepare_forward_input(
        engine_.args_, engine_.threadpool_.get())));
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

  for (int32_t dp_rank = 0; dp_rank < engine_.dp_size_; ++dp_rank) {
    batched_inputs[dp_rank].dp_global_token_nums = dp_global_token_nums;
    batched_inputs[dp_rank].dp_is_decode = dp_is_decode;
    batched_inputs[dp_rank].global_empty_kv_cache = global_empty_kv_cache;
    if (batched_inputs[dp_rank].batch_forward_type.is_empty()) {
      batched_inputs[dp_rank].batch_forward_type = batch_forward_type;
    }
  }

  return batched_inputs;
}

ForwardOutput RecEngine::LlmRecEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.worker_clients_.empty()) {
    return {};
  }

  DCHECK(engine_.dp_size_ == static_cast<int32_t>(batches.size()))
      << "Split DP batch failed with dp_size as " << engine_.dp_size_
      << " and actual batch size as " << batches.size() << ".";

  auto run_one_step = [this, &batches](int step_idx) -> bool {
    Timer timer;
    auto raw_forward_inputs = prepare_inputs(batches);
    COUNTER_ADD(prepare_input_latency_microseconds,
                timer.elapsed_microseconds());

    const bool all_empty =
        std::all_of(raw_forward_inputs.begin(),
                    raw_forward_inputs.end(),
                    [](const RawForwardInput& input) {
                      return input.flatten_tokens_vec.empty();
                    });
    if (all_empty) {
      return false;
    }

    std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
    futures.reserve(engine_.worker_clients_num_);

    timer.reset();
    for (size_t worker_rank = 0; worker_rank < engine_.worker_clients_num_;
         ++worker_rank) {
      auto dp_rank = worker_rank / engine_.dp_local_tp_size_;
      futures.emplace_back(engine_.worker_clients_[worker_rank]->step_async(
          raw_forward_inputs[dp_rank]));
    }
    auto results = folly::collectAll(futures).get();

    if (step_idx == 0) {
      COUNTER_ADD(rec_first_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (step_idx == 1) {
      COUNTER_ADD(rec_second_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (step_idx == 2) {
      COUNTER_ADD(rec_third_token_latency_microseconds,
                  timer.elapsed_microseconds());
    }

    timer.reset();
    size_t dp_rank = 0;
    for (size_t worker_rank = 0; worker_rank < engine_.worker_clients_num_;
         worker_rank += engine_.dp_local_tp_size_) {
      auto result = results[worker_rank].value();
      if (!result.has_value()) {
        LOG(FATAL) << "Failed to execute model, result has no value";
      }
      if (result.value().src_seq_idxes.empty()) {
        batches[dp_rank].process_sample_output(result.value(), false);
      } else {
        batches[dp_rank].process_beam_search_output(result.value(), false);
        // Transfer src_blocks_ to blocks_ for beam search sequences
        // RecEngine doesn't have Scheduler/BlockManagerPool to trigger this
        for (size_t i = 0; i < batches[dp_rank].size(); ++i) {
          auto* seq = batches[dp_rank][i];
          if (seq->check_beam_search() &&
              !seq->kv_state().src_blocks().empty()) {
            seq->kv_state().process_beam_search(std::nullopt);
          }
        }
      }
      // Refresh sequences_ from sequence_groups_ after beam search processing.
      // This is needed because SequencesGroup::process_beam_search() replaces
      // its internal sequences_, invalidating pointers in Batch::sequences_.
      batches[dp_rank].refresh_sequences_from_groups();
      ++dp_rank;
    }
    COUNTER_ADD(rec_sampling_latency_microseconds,
                timer.elapsed_microseconds());
    return true;
  };

  // Get dynamic max steps from batch (based on max_tokens in requests)
  const size_t max_steps = get_max_steps_from_batch(batches);

  for (size_t step_idx = 0; step_idx < max_steps; ++step_idx) {
    if (!run_one_step(step_idx)) {
      break;
    }
  }

  for (auto& batch : batches) {
    batch.finish();
  }
  return {};
}

std::vector<int64_t>
RecEngine::LlmRecEnginePipeline::get_active_activation_memory() const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(futures.size());
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

size_t RecEngine::LlmRecEnginePipeline::get_max_steps_from_batch(
    std::vector<Batch>& batches) const {
  size_t max_steps = 0;
  bool has_stopping_checker = false;
  for (auto& batch : batches) {
    // Use get_sequences() to handle both sequences_ and sequence_groups_
    // This ensures compatibility with both LlmRec and OneRec scenarios
    auto sequences = batch.get_sequences();
    for (auto* seq : sequences) {
      const auto* stopping_checker = seq->stopping_checker();
      if (stopping_checker) {
        has_stopping_checker = true;
        max_steps =
            std::max(max_steps, stopping_checker->get_max_generated_tokens());
      }
    }
  }
  // If has stopping_checker, use max_tokens from it;
  // otherwise fall back to kRecDecodeSteps for OneRec compatibility
  if (has_stopping_checker && max_steps > 0) {
    return max_steps;
  }
  return kRecDecodeSteps;
}

// ============================================================
// OneRecEnginePipeline Implementation
// ============================================================

RecEngine::OneRecEnginePipeline::OneRecEnginePipeline(RecEngine& engine)
    : RecEnginePipeline(engine) {}

void RecEngine::OneRecEnginePipeline::setup_workers() {
  // OneRec uses local workers, no DistManager setup needed
}

void RecEngine::OneRecEnginePipeline::process_group_test() {
  if (engine_.workers_.size() > 1) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(engine_.workers_.size());
    for (auto& worker : engine_.workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    const int timeout_seconds = util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
}

bool RecEngine::OneRecEnginePipeline::init_model_workers(
    const std::string& model_path) {
  const auto& devices = engine_.options_.devices();
  if (devices.size() > 1) {
    engine_.process_groups_ =
        parallel_state::create_npu_process_groups(devices);
  }

  engine_.workers_.clear();
  WorkerType worker_type = WorkerType::REC;
  const int32_t world_size = static_cast<int32_t>(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg =
        world_size > 1 ? engine_.process_groups_[i].get() : nullptr;
    ParallelArgs parallel_args(rank, world_size, pg);
    engine_.workers_.emplace_back(std::make_unique<Worker>(
        parallel_args, devices[i], engine_.options_, worker_type));
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.push_back(worker->init_model_async(model_path, FLAGS_random_seed));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

int64_t RecEngine::OneRecEnginePipeline::estimate_min_available_memory() {
  const int64_t max_cache_size = engine_.options_.max_cache_size();
  const double max_memory_utilization =
      engine_.options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.push_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for worker: " << i;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
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
  return cache_size_in_bytes;
}

bool RecEngine::OneRecEnginePipeline::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

size_t RecEngine::OneRecEnginePipeline::num_workers() const {
  return engine_.workers_.size();
}

ForwardOutput RecEngine::OneRecEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.workers_.empty()) {
    return {};
  }

  Timer timer;
  // OneRec does not need refresh_forward_type
  auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_microseconds, timer.elapsed_microseconds());

  if (!forward_inputs.token_ids.defined()) {
    return {};
  }

  timer.reset();
  const auto& prefill_output = get_model_output(forward_inputs);
  COUNTER_ADD(rec_first_token_latency_microseconds,
              timer.elapsed_microseconds());

  timer.reset();
  batches[0].process_sample_output(prefill_output.sample_output, false);
  COUNTER_ADD(rec_sampling_latency_microseconds, timer.elapsed_microseconds());

  ForwardOutput decode_output;
  for (size_t i = 0; i < kRecDecodeSteps; ++i) {
    timer.reset();
    // OneRec does not need refresh_forward_type
    forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
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
  return decode_output;
}

ForwardOutput RecEngine::OneRecEnginePipeline::get_model_output(
    const ForwardInput& model_inputs) {
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->step_async(model_inputs));
  }
  auto results = folly::collectAll(futures).get();
  auto forward_output = results.front().value();

  CHECK(forward_output.has_value()) << "Failed to execute model";
  return forward_output.value();
}

std::vector<int64_t>
RecEngine::OneRecEnginePipeline::get_active_activation_memory() const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(futures.size());
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

// ============================================================
// RecEngine pipeline factory (static method)
// ============================================================
std::unique_ptr<RecEngine::RecEnginePipeline> RecEngine::create_pipeline(
    RecPipelineType type,
    RecEngine& engine) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecEnginePipeline>(engine);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecEnginePipeline>(engine);
    default:
      LOG(FATAL) << "Unknown RecEngine pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
