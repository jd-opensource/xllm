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
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/params_utils.h"
#include "util/hash_util.h"

namespace xllm {
RemoteWorker::RemoteWorker(int32_t global_rank,
                           const std::string& server_address,
                           const torch::Device& d)
    : global_rank_(global_rank), device_(d) {
  // Initialize brpc channel
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;
  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc channel";
    return;
  }
  // Initialize stub
  stub_.reset(new proto::DistributeWorker_Stub(&channel_));

  wait_for_server_ready(server_address);
}

bool RemoteWorker::wait_for_server_ready(const std::string& server_address) {
  proto::Status req;
  proto::Status resp;

  // Retry until server initialize ready
  int try_count = 0;
  brpc::Controller cntl;
  while (try_count < FLAGS_max_connect_count) {
    cntl.Reset();
    stub_->Hello(&cntl, &req, &resp, nullptr);
    if (cntl.Failed() || !resp.ok()) {
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_sleep_time_second));
    } else {
      LOG(INFO) << "RemoteWorker Hello connected, server_address: "
                << server_address << ", global_rank_: " << global_rank_;
      break;
    }

    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "RemoteWorker Hello method failed, global_rank_ is "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }

  return true;
}

bool RemoteWorker::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  proto::KVCacheShape shape;
  shape.mutable_key_shape()->Reserve(kv_cache_shape[0].size());
  shape.mutable_value_shape()->Reserve(kv_cache_shape[1].size());
  for (int32_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    shape.add_key_shape(kv_cache_shape[0][i]);
    shape.add_value_shape(kv_cache_shape[1][i]);
  }
  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateKVCache(&cntl, &shape, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "allocate_kv_cache failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool RemoteWorker::allocate_continuous_kv_cache(
    const std::vector<XTensor::Options>& options) {
  proto::XTensorOptionsVec xtensor_options_vec;
  xtensor_options_vec.mutable_key_options()->set_num_kv_heads(
      options[0].num_kv_heads());
  xtensor_options_vec.mutable_key_options()->set_head_size(
      options[0].head_size());
  xtensor_options_vec.mutable_key_options()->set_max_context_len(
      options[0].max_context_len());
  xtensor_options_vec.mutable_key_options()->set_max_seqs_per_batch(
      options[0].max_seqs_per_batch());
  xtensor_options_vec.mutable_value_options()->set_num_kv_heads(
      options[1].num_kv_heads());
  xtensor_options_vec.mutable_value_options()->set_head_size(
      options[1].head_size());
  xtensor_options_vec.mutable_value_options()->set_max_context_len(
      options[1].max_context_len());
  xtensor_options_vec.mutable_value_options()->set_max_seqs_per_batch(
      options[1].max_seqs_per_batch());

  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateContinuousKVCache(&cntl, &xtensor_options_vec, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "allocate_continuous_kv_cache failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

void RemoteWorker::get_device_info(std::string& device_ip, uint16_t& port) {
  proto::Empty req;
  proto::DeviceInfo resp;
  brpc::Controller cntl;
  stub_->GetDeviceInfo(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetDeviceInfo failed." << cntl.ErrorText();
    return;
  }
  device_ip = resp.device_ip();
  port = resp.listen_port();
}

void RemoteWorker::get_cache_info(uint64_t& cluster_id,
                                  std::string& addr,
                                  int64_t& k_cache_id,
                                  int64_t& v_cache_id) {
  proto::Empty req;
  proto::CacheInfo resp;
  brpc::Controller cntl;
  stub_->GetCacheInfo(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetCacheInfo failed, " << cntl.ErrorText();
    return;
  }
  cluster_id = resp.cluster_id();
  addr = resp.addr();
  k_cache_id = resp.k_cache_id();
  v_cache_id = resp.v_cache_id();
}

bool RemoteWorker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<std::string>& device_ips,
                                const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_device_ips()->Reserve(device_ips.size());
  cluster_info.mutable_ports()->Reserve(ports.size());
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_device_ips(device_ips[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->LinkCluster(&cntl, &cluster_info, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(INFO) << "LinkCluster failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool RemoteWorker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<std::string>& device_ips,
                                  const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_device_ips()->Reserve(device_ips.size());
  cluster_info.mutable_ports()->Reserve(ports.size());
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_device_ips(device_ips[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->UnlinkCluster(&cntl, &cluster_info, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(INFO) << "UnlinkCluster failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool RemoteWorker::init_model(const std::string& model_weights_path) {
  proto::ModelPath request;
  request.set_model_weights_path(model_weights_path);
  proto::Status response;
  brpc::Controller cntl;
  stub_->InitModel(&cntl, &request, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "init_model failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

std::tuple<int64_t, int64_t> RemoteWorker::estimate_kv_cache_capacity() {
  proto::Empty req;
  proto::DeviceMemory mem;
  brpc::Controller cntl;
  stub_->ProfileDeviceMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "estimate_kv_cache_capacity failed: " << cntl.ErrorText();
  }
  std::tuple<int64_t, int64_t> result(mem.available_memory(),
                                      mem.total_memory());
  return result;
}

bool RemoteWorker::pull_kv_blocks(const uint64_t src_cluster_id,
                                  const std::string& src_addr,
                                  const int64_t src_k_cache_id,
                                  const int64_t src_v_cache_id,
                                  const std::vector<uint64_t>& src_blocks,
                                  const std::vector<uint64_t>& dst_blocks) {
  proto::PullKVCacheRequest request;
  request.set_cluster_id(src_cluster_id);
  request.set_addr(src_addr);
  request.set_k_cache_id(src_k_cache_id);
  request.set_v_cache_id(src_v_cache_id);
  ADD_VECTOR_TO_PROTO(request.mutable_src_blocks(), src_blocks);
  ADD_VECTOR_TO_PROTO(request.mutable_dst_blocks(), dst_blocks);
  proto::Status s;
  brpc::Controller cntl;
  stub_->PullKVCache(&cntl, &request, &s, nullptr);

  return s.ok();
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
    stub_->ProfileDeviceMemory(&cntl, &req, &mem, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "estimate_kv_cache_capacity_async failed: "
                 << cntl.ErrorText();
    }
    std::tuple<int64_t, int64_t> result(mem.available_memory(),
                                        mem.total_memory());
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
    const RawForwardInput& inputs) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, inputs = inputs, promise = std::move(promise)]() mutable {
        // 1. convert to proto::ForwardInput
        proto::ForwardInput pb_forward_input;
        forward_input_to_proto(inputs, &pb_forward_input);

        // 2. call ExecuteModel with callback
        auto done = new ExecuteModelClosure();
        done->promise = std::move(promise);
        stub_->ExecuteModel(
            &done->cntl, &pb_forward_input, &done->pb_output, done);
      });

  return future;
}

void ExecuteModelClosure::Run() {
  std::unique_ptr<ExecuteModelClosure> self_guard(this);

  if (cntl.Failed()) {
    LOG(ERROR) << "Execute_model_async failed. Error code : "
               << cntl.ErrorCode() << ", error message : " << cntl.ErrorText();
  }

  // 3. parse tokens
  RawForwardOutput raw_forward_output;
  proto_to_forward_output(pb_output, raw_forward_output);
  promise.setValue(raw_forward_output);

  return;
}

folly::SemiFuture<folly::Unit> RemoteWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::Status s;
    brpc::Controller cntl;
    stub_->ProcessGroupTest(&cntl, &req, &s, nullptr);
    if (cntl.Failed() || !s.ok()) {
      LOG(ERROR) << "process_group_test_async failed, " << cntl.ErrorText();
    }
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
        auto done = new InitModelClosure();
        done->promise = std::move(promise);
        proto::ModelPath request;
        request.set_model_weights_path(model_weights_path);
        stub_->InitModel(&done->cntl, &request, &done->response, done);
      });
  return future;
}

void InitModelClosure::Run() {
  std::unique_ptr<InitModelClosure> self_guard(this);

  bool success = !cntl.Failed() && response.ok();
  if (!success) {
    LOG(ERROR) << "Init_model_async failed, " << cntl.ErrorText();
  } else {
    LOG(INFO) << "Init_model_async succeed.";
  }
  promise.setValue(success);

  return;
}

folly::SemiFuture<bool> RemoteWorker::allocate_kv_cache_async(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        proto::KVCacheShape shape;
        shape.mutable_key_shape()->Reserve(kv_cache_shape[0].size());
        shape.mutable_value_shape()->Reserve(kv_cache_shape[1].size());
        for (int32_t i = 0; i < kv_cache_shape[0].size(); ++i) {
          shape.add_key_shape(kv_cache_shape[0][i]);
          shape.add_value_shape(kv_cache_shape[1][i]);
        }
        proto::Status s;
        brpc::Controller cntl;
        stub_->AllocateKVCache(&cntl, &shape, &s, nullptr);
        if (cntl.Failed() || !s.ok()) {
          LOG(ERROR) << "allocate_kv_cache_async failed, " << cntl.ErrorText();
          promise.setValue(false);
        } else {
          promise.setValue(s.ok());
        }
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_continuous_kv_cache_async(
    const std::vector<XTensor::Options>& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    proto::XTensorOptionsVec xtensor_options_vec;
    xtensor_options_vec.mutable_key_options()->set_num_kv_heads(
        options[0].num_kv_heads());
    xtensor_options_vec.mutable_key_options()->set_head_size(
        options[0].head_size());
    xtensor_options_vec.mutable_key_options()->set_max_context_len(
        options[0].max_context_len());
    xtensor_options_vec.mutable_key_options()->set_max_seqs_per_batch(
        options[0].max_seqs_per_batch());
    xtensor_options_vec.mutable_value_options()->set_num_kv_heads(
        options[1].num_kv_heads());
    xtensor_options_vec.mutable_value_options()->set_head_size(
        options[1].head_size());
    xtensor_options_vec.mutable_value_options()->set_max_context_len(
        options[1].max_context_len());
    xtensor_options_vec.mutable_value_options()->set_max_seqs_per_batch(
        options[1].max_seqs_per_batch());
    proto::Status s;
    brpc::Controller cntl;
    stub_->AllocateContinuousKVCache(&cntl, &xtensor_options_vec, &s, nullptr);
    if (cntl.Failed() || !s.ok()) {
      LOG(ERROR) << "allocate_continuous_kv_cache_async failed, "
                 << cntl.ErrorText();
      promise.setValue(false);
    } else {
      promise.setValue(s.ok());
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
    proto::AllocateKVCacheWithTransferRequest request;
    request.set_kv_cache_size(kv_cache_size);
    request.mutable_kv_cache_shape()->mutable_key_shape()->Reserve(
        kv_cache_shape[0].size());
    request.mutable_kv_cache_shape()->mutable_value_shape()->Reserve(
        kv_cache_shape[1].size());
    for (int32_t i = 0; i < kv_cache_shape[0].size(); ++i) {
      request.mutable_kv_cache_shape()->add_key_shape(kv_cache_shape[0][i]);
      request.mutable_kv_cache_shape()->add_value_shape(kv_cache_shape[1][i]);
    }
    proto::Status s;
    brpc::Controller cntl;
    stub_->AllocateKVCacheWithTransfer(&cntl, &request, &s, nullptr);
    if (cntl.Failed() || !s.ok()) {
      LOG(ERROR) << "AllocateKVCacheWithTransfer failed, " << cntl.ErrorText();
      promise.setValue(false);
    } else {
      promise.setValue(s.ok());
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
    proto::PullKVCacheRequest request;
    request.set_cluster_id(src_cluster_id);
    request.set_addr(src_addr);
    request.set_k_cache_id(src_k_cache_id);
    request.set_v_cache_id(src_v_cache_id);
    ADD_VECTOR_TO_PROTO(request.mutable_src_blocks(), src_blocks);
    ADD_VECTOR_TO_PROTO(request.mutable_dst_blocks(), dst_blocks);
    proto::Status s;
    brpc::Controller cntl;
    stub_->PullKVCache(&cntl, &request, &s, nullptr);
    if (cntl.Failed() || !s.ok()) {
      LOG(ERROR) << "PullKVCache failed, " << cntl.ErrorText();
      promise.setValue(false);
    } else {
      promise.setValue(s.ok());
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
    proto::CacheBlockInfos pb_cache_block_info;
    if (!cache_block_info_to_proto(cache_block_info, &pb_cache_block_info)) {
      promise.setValue(0);
      return;
    }

    auto done = new LoadKVCacheFromStoreClosure();
    done->promise = std::move(promise);
    stub_->LoadKVCacheFromStore(
        &done->cntl, &pb_cache_block_info, &done->response, done);
  });
  return future;
}

void LoadKVCacheFromStoreClosure::Run() {
  std::unique_ptr<LoadKVCacheFromStoreClosure> self_guard(this);

  bool success = !cntl.Failed();
  if (!success) {
    promise.setValue(0);
  } else {
    promise.setValue(response.success_cnt());
  }
  return;
}

const torch::Device& RemoteWorker::device() const {
  LOG(ERROR) << "RemoteWorker Method device is UnImplemented.";
}

folly::SemiFuture<std::optional<RawForwardOutput>>
RemoteWorker::get_last_step_result_async() {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::ForwardOutput pb_output;
    brpc::Controller cntl;
    stub_->GetLastStepResult(&cntl, &req, &pb_output, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Get last step model output result failed, "
                 << cntl.ErrorText();
    }

    // parse tokens
    RawForwardOutput raw_forward_output;
    proto_to_forward_output(pb_output, raw_forward_output);
    promise.setValue(std::move(raw_forward_output));
  });
  return future;
}

int64_t RemoteWorker::get_active_activation_memory() {
  proto::Empty req;
  proto::ActivationMemory mem;
  brpc::Controller cntl;
  stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "get_active_activation_memory failed: " << cntl.ErrorText();
  }
  return mem.active_activation_memory();
}

folly::SemiFuture<int64_t> RemoteWorker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::ActivationMemory mem;
    brpc::Controller cntl;
    stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "get_active_activation_memory_async failed: "
                 << cntl.ErrorText();
    }
    promise.setValue(mem.active_activation_memory());
  });
  return future;
}

}  // namespace xllm
