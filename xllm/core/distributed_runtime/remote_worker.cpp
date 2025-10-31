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

#include "remote_worker.h"

#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <optional>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/params_utils.h"
#include "util/hash_util.h"

namespace xllm {
RemoteWorker::RemoteWorker(int32_t global_rank,
                           const std::string& server_address,
                           const torch::Device& d,
                           std::unique_ptr<CommChannel> channel)
    : global_rank_(global_rank), device_(d), channel_(std::move(channel)) {
  wait_for_server_ready(server_address);
}

bool RemoteWorker::wait_for_server_ready(const std::string& server_address) {
  // Retry until server initialize ready
  int try_count = 0;
  const int sleep_time_second = 3;
  while (try_count < FLAGS_max_reconnect_count) {
    if (channel_->hello()) {
      LOG(INFO) << "RemoteWorker Hello connected, server_address: "
                << server_address << ", global_rank_: " << global_rank_;
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    }

    try_count++;
  }

  if (try_count >= FLAGS_max_reconnect_count) {
    LOG(ERROR) << "RemoteWorker Hello method failed, global_rank_ is "
               << global_rank_;
    return false;
  }

  return true;
}

bool RemoteWorker::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return channel_->allocate_kv_cache(kv_cache_shape);
}

bool RemoteWorker::allocate_continuous_kv_cache(
    const std::vector<XTensor::Options>& options) {
  return channel_->allocate_continuous_kv_cache(options);
}

void RemoteWorker::get_device_info(std::string& device_ip, uint16_t& port) {
  channel_->get_device_info(device_ip, port);
}

void RemoteWorker::get_cache_info(uint64_t& cluster_id,
                                  std::string& addr,
                                  int64_t& k_cache_id,
                                  int64_t& v_cache_id) {
  channel_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
}

bool RemoteWorker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<std::string>& device_ips,
                                const std::vector<uint16_t>& ports) {
  return channel_->link_cluster(cluster_ids, addrs, device_ips, ports);
}

bool RemoteWorker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<std::string>& device_ips,
                                  const std::vector<uint16_t>& ports) {
  return channel_->unlink_cluster(cluster_ids, addrs, device_ips, ports);
}

bool RemoteWorker::init_model(const std::string& model_weights_path) {
  return channel_->init_model(model_weights_path);
}

std::tuple<int64_t, int64_t> RemoteWorker::estimate_kv_cache_capacity() {
  proto::Empty req;
  proto::DeviceMemory mem;
  brpc::Controller cntl;
  int64_t available_memory = 0;
  int64_t total_memory = 0;

  channel_->estimate_kv_cache_capacity(available_memory, total_memory);
  std::tuple<int64_t, int64_t> result(available_memory, total_memory);
  return result;
}

bool RemoteWorker::pull_kv_blocks(const uint64_t src_cluster_id,
                                  const std::string& src_addr,
                                  const int64_t src_k_cache_id,
                                  const int64_t src_v_cache_id,
                                  const std::vector<uint64_t>& src_blocks,
                                  const std::vector<uint64_t>& dst_blocks) {
  return channel_->pull_kv_blocks(src_cluster_id,
                                  src_addr,
                                  src_k_cache_id,
                                  src_v_cache_id,
                                  src_blocks,
                                  dst_blocks);
}

ForwardInput RemoteWorker::prepare_inputs(Batch& batch) {
  LOG(ERROR) << "RemoteWorker Method prepare_inputs is UnImplemented.";
}

std::optional<ForwardOutput> RemoteWorker::step(const ForwardInput& inputs) {
  LOG(ERROR) << "RemoteWorker Method step is UnImplemented.";
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
RemoteWorker::estimate_kv_cache_capacity_async() {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::DeviceMemory mem;
    brpc::Controller cntl;
    int64_t available_memory = 0;
    int64_t total_memory = 0;

    channel_->estimate_kv_cache_capacity(available_memory, total_memory);
    std::tuple<int64_t, int64_t> result(available_memory, total_memory);
    promise.setValue(result);
  });
  return future;
}

folly::SemiFuture<std::optional<ForwardOutput>> RemoteWorker::step_async(
    const ForwardInput& inputs) {
  LOG(ERROR) << "RemoteWorker Method step_async with "
                "ForwardInput param is UnImplemented.";
}

folly::SemiFuture<std::optional<RawForwardOutput>> RemoteWorker::step_async(
    const std::vector<RawForwardInput>& inputs) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, inputs = inputs, promise = std::move(promise)]() mutable {
        channel_->execute_model_async(inputs, promise);
      });

  return future;
}

folly::SemiFuture<folly::Unit> RemoteWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->process_group_test();
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::init_model_async(
    const std::string& model_weights_path) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_weights_path, promise = std::move(promise)]() mutable {
        // call InitModel with callback
        channel_->init_model_async(model_weights_path, promise);
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_kv_cache_async(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        if (!channel_->allocate_kv_cache(kv_cache_shape)) {
          LOG(ERROR) << "allocate_kv_cache_async failed";
          promise.setValue(false);
        } else {
          promise.setValue(true);
        }
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_continuous_kv_cache_async(
    const std::vector<XTensor::Options>& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    if (!channel_->allocate_continuous_kv_cache(options)) {
      LOG(ERROR) << "allocate_continuous_kv_cache_async failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_kv_cache_with_transfer_async(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        kv_cache_size,
                        kv_cache_shape,
                        promise = std::move(promise)]() mutable {
    if (!channel_->allocate_kv_cache_with_transfer(kv_cache_size,
                                                   kv_cache_shape)) {
      LOG(ERROR) << "AllocateKVCacheWithTransfer failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        src_k_cache_id,
                        src_v_cache_id,
                        &src_blocks,
                        &dst_blocks,
                        promise = std::move(promise)]() mutable {
    if (!channel_->pull_kv_blocks(src_cluster_id,
                                  src_addr,
                                  src_k_cache_id,
                                  src_v_cache_id,
                                  src_blocks,
                                  dst_blocks)) {
      LOG(ERROR) << "PullKVCache failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<uint32_t> RemoteWorker::load_kv_blocks_from_store_async(
    const std::vector<CacheBlockInfo> cache_block_info) {
  folly::Promise<uint32_t> promise;
  auto future = promise.getSemiFuture();
  general_threadpool_.schedule([this,
                                cache_block_info = std::move(cache_block_info),
                                promise = std::move(promise)]() mutable {
    channel_->load_kv_blocks_from_store_async(cache_block_info, promise);
  });
  return future;
}

const torch::Device& RemoteWorker::device() const {
  LOG(ERROR) << "RemoteWorker Method device is UnImplemented.";
}

folly::SemiFuture<std::optional<RawForwardOutput>>
RemoteWorker::get_last_step_result_async() {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->get_last_step_result_async(promise);
  });
  return future;
}

int64_t RemoteWorker::get_active_activation_memory() {
  int64_t memory = 0;
  channel_->get_active_activation_memory(memory);
  return memory;
}

folly::SemiFuture<int64_t> RemoteWorker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->get_active_activation_memory_async(promise);
  });
  return future;
}

}  // namespace xllm
