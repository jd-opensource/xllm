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

#include "worker.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/embed_vlm_worker_impl.h"
#include "runtime/embed_worker_impl.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/rec_worker_impl.h"
#include "runtime/speculative_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/timer.h"

namespace xllm {
Worker::Worker(const ParallelArgs& parallel_args,
               const torch::Device& device,
               const runtime::Options& options,
               WorkerType worker_type) {
  if (options.enable_speculative_decode()) {
    impl_ = new SpeculativeWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::LLM) {
    impl_ = new LLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ = new VLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::ELM) {
    impl_ = new EmbedWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::EVLM) {
    impl_ = new EmbedVLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::REC) {
    impl_ = new RecWorkerImpl(parallel_args, device, options);
  } else {
    LOG(ERROR) << "Unknown worker type, please check logic";
  }
}

Worker::~Worker() { delete impl_; }

bool Worker::init_model(const std::string& model_weights_path) {
  return impl_->init_model(model_weights_path);
}

bool Worker::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

bool Worker::allocate_continuous_kv_cache(
    const std::vector<XTensor::Options>& options) {
  return impl_->allocate_continuous_kv_cache(options);
}

void Worker::get_device_info(std::string& device_ip, uint16_t& port) {
  impl_->get_device_info(device_ip, port);
}

void Worker::get_cache_info(uint64_t& cluster_id,
                            std::string& addr,
                            int64_t& k_cache_id,
                            int64_t& v_cache_id) {
  impl_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
}

bool Worker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                          const std::vector<std::string>& addrs,
                          const std::vector<std::string>& device_ips,
                          const std::vector<uint16_t>& ports) {
  return impl_->link_cluster(cluster_ids, addrs, device_ips, ports);
}

bool Worker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports) {
  return impl_->unlink_cluster(cluster_ids, addrs, device_ips, ports);
}

std::tuple<int64_t, int64_t> Worker::estimate_kv_cache_capacity() {
  return impl_->estimate_kv_cache_capacity();
}

ForwardInput Worker::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

std::optional<ForwardOutput> Worker::step(const ForwardInput& inputs) {
  // TODO to adapt multi stream parallel later
  BatchedForwardInputs batched_inputs;
  batched_inputs.micro_inputs = {std::move(inputs)};
  return impl_->step(batched_inputs);
}

const bool Worker::is_driver() { return impl_->is_driver(); }

folly::SemiFuture<std::tuple<int64_t, int64_t>>
Worker::estimate_kv_cache_capacity_async() {
  return impl_->estimate_kv_cache_capacity_async();
}

folly::SemiFuture<std::optional<ForwardOutput>> Worker::step_async(
    const BatchedForwardInputs& inputs) {
  return impl_->step_async(inputs);
}

folly::SemiFuture<folly::Unit> Worker::process_group_test_async() {
  return impl_->process_group_test_async();
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> Worker::init_model_async(
    const std::string& model_weights_path) {
  return impl_->init_model_async(model_weights_path);
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_async(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache_async(kv_cache_shape);
}

folly::SemiFuture<bool> Worker::allocate_continuous_kv_cache_async(
    const std::vector<XTensor::Options>& options) {
  return impl_->allocate_continuous_kv_cache_async(options);
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_with_transfer_async(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer_async(kv_cache_size,
                                                      kv_cache_shape);
}

folly::SemiFuture<bool> Worker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  return impl_->pull_kv_blocks_async(src_cluster_id,
                                     src_addr,
                                     src_k_cache_id,
                                     src_v_cache_id,
                                     src_blocks,
                                     dst_blocks);
}

folly::SemiFuture<uint32_t> Worker::load_kv_blocks_from_store_async(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  return impl_->load_kv_blocks_from_store_async(cache_block_info);
}

const torch::Device& Worker::device() const { return impl_->device(); }

folly::SemiFuture<std::optional<ForwardOutput>>
Worker::get_last_step_result_async() {
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_last_step_result());
  });
  return future;
}

int64_t Worker::get_active_activation_memory() {
  return impl_->get_active_activation_memory();
}

folly::SemiFuture<int64_t> Worker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_active_activation_memory());
  });
  return future;
}

}  // namespace xllm
