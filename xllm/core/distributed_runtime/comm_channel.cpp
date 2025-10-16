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

#include "comm_channel.h"

#include <brpc/controller.h>
#include <glog/logging.h>

namespace xllm {

bool CommChannel::init_brpc(const std::string& server_address) {
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;

  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc Channel";
    return false;
  }

  stub_.reset(new proto::DistributeWorker_Stub(&channel_));
  return true;
}

bool CommChannel::hello() {
  proto::Status req;
  proto::Status resp;
  brpc::Controller cntl;

  cntl.Reset();
  stub_->Hello(&cntl, &req, &resp, nullptr);
  if (cntl.Failed() || !resp.ok()) {
    LOG(ERROR) << "Hello request failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  proto::KVCacheShape shape;
  shape.mutable_key_shape()->Reserve(kv_cache_shape[0].size());
  shape.mutable_value_shape()->Reserve(kv_cache_shape[1].size());

  for (size_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    shape.add_key_shape(kv_cache_shape[0][i]);
    shape.add_value_shape(kv_cache_shape[1][i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateKVCache(&cntl, &shape, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "allocate_kv_cache failed: " << cntl.ErrorText();
    return false;
  }

  return true;
}

bool CommChannel::allocate_continuous_kv_cache(
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
    LOG(ERROR) << "allocate_continuous_kv_cache failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::get_device_info(std::string& device_ip, uint16_t& port) {
  proto::Empty req;
  proto::DeviceInfo resp;
  brpc::Controller cntl;

  stub_->GetDeviceInfo(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetDeviceInfo failed: " << cntl.ErrorText();
    return false;
  }

  device_ip = resp.device_ip();
  port = resp.listen_port();
  return true;
}

bool CommChannel::get_cache_info(uint64_t& cluster_id,
                                 std::string& addr,
                                 int64_t& k_cache_id,
                                 int64_t& v_cache_id) {
  proto::Empty req;
  proto::CacheInfo resp;
  brpc::Controller cntl;

  stub_->GetCacheInfo(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetCacheInfo failed: " << cntl.ErrorText();
    return false;
  }

  cluster_id = resp.cluster_id();
  addr = resp.addr();
  k_cache_id = resp.k_cache_id();
  v_cache_id = resp.v_cache_id();
  return true;
}

bool CommChannel::link_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<std::string>& device_ips,
                               const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_device_ips()->Reserve(device_ips.size());
  cluster_info.mutable_ports()->Reserve(ports.size());

  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_device_ips(device_ips[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->LinkCluster(&cntl, &cluster_info, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "LinkCluster failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                 const std::vector<std::string>& addrs,
                                 const std::vector<std::string>& device_ips,
                                 const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_device_ips()->Reserve(device_ips.size());
  cluster_info.mutable_ports()->Reserve(ports.size());

  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_device_ips(device_ips[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->UnlinkCluster(&cntl, &cluster_info, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "UnlinkCluster failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::init_model(const std::string& model_weights_path) {
  proto::ModelPath request;

  request.set_model_weights_path(model_weights_path);
  proto::Status response;
  brpc::Controller cntl;
  stub_->InitModel(&cntl, &request, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "init_model failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::init_model_async(const std::string& model_weights_path,
                                   folly::Promise<bool>& promise) {
  proto::ModelPath request;

  request.set_model_weights_path(model_weights_path);
  auto done = new InitModelClosure();
  done->promise = std::move(promise);
  stub_->InitModel(&done->cntl, &request, &done->response, done);

  return true;
}

bool CommChannel::estimate_kv_cache_capacity(int64_t& available_memory,
                                             int64_t& total_memory) {
  proto::Empty req;
  proto::DeviceMemory mem;
  brpc::Controller cntl;

  stub_->ProfileDeviceMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "estimate_kv_cache_capacity failed: " << cntl.ErrorText();
    return false;
  }

  available_memory = mem.available_memory();
  total_memory = mem.total_memory();
  return true;
}

bool CommChannel::pull_kv_blocks(const uint64_t src_cluster_id,
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

  return !cntl.Failed() && s.ok();
}

void CommChannel::execute_model_async(
    const std::vector<RawForwardInput>& inputs,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_model_with_brpc(inputs, promise);
}

bool CommChannel::process_group_test() {
  proto::Empty req;
  proto::Status s;
  brpc::Controller cntl;

  stub_->ProcessGroupTest(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "process_group_test failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::allocate_kv_cache_with_transfer(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  proto::AllocateKVCacheWithTransferRequest request;
  request.set_kv_cache_size(kv_cache_size);

  auto* shape = request.mutable_kv_cache_shape();
  shape->mutable_key_shape()->Reserve(kv_cache_shape[0].size());
  shape->mutable_value_shape()->Reserve(kv_cache_shape[1].size());

  for (size_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    shape->add_key_shape(kv_cache_shape[0][i]);
    shape->add_value_shape(kv_cache_shape[1][i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateKVCacheWithTransfer(&cntl, &request, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "AllocateKVCacheWithTransfer failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::load_kv_blocks_from_store_async(
    const std::vector<CacheBlockInfo>& cache_block_info,
    folly::Promise<uint32_t>& promise) {
  proto::CacheBlockInfos pb_cache_block_info;
  if (!cache_block_info_to_proto(cache_block_info, &pb_cache_block_info)) {
    promise.setValue(0);
    return false;
  }

  auto done = new LoadKVCacheFromStoreClosure();
  done->promise = std::move(promise);
  stub_->LoadKVCacheFromStore(
      &done->cntl, &pb_cache_block_info, &done->response, done);

  return true;
}

bool CommChannel::get_last_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::Empty req;
  proto::ForwardOutput pb_output;
  brpc::Controller cntl;
  stub_->GetLastStepResult(&cntl, &req, &pb_output, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Get last step model output result failed, "
               << cntl.ErrorText();
    return false;
  }

  // parse tokens
  RawForwardOutput raw_forward_output;
  proto_to_forward_output(pb_output, raw_forward_output);
  promise.setValue(std::move(raw_forward_output));

  return true;
}

bool CommChannel::get_active_activation_memory(int64_t& memory) {
  proto::Empty req;
  proto::ActivationMemory mem;
  brpc::Controller cntl;

  stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetActiveActivationMemory failed: " << cntl.ErrorText();
    return false;
  }

  memory = mem.active_activation_memory();
  return true;
}

bool CommChannel::get_active_activation_memory_async(
    folly::Promise<int64_t>& promise) {
  proto::Empty req;
  proto::ActivationMemory mem;
  brpc::Controller cntl;

  stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "get_active_activation_memory_async failed: "
               << cntl.ErrorText();
    promise.setValue(0);
    return false;
  }
  promise.setValue(mem.active_activation_memory());
  return true;
}

bool CommChannel::execute_model_with_brpc(
    const std::vector<RawForwardInput>& inputs,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  // convert to proto::BatchedForwardInputs
  proto::BatchedForwardInputs pb_batched_fwd_inputs;
  std::vector<proto::ForwardInput> batched_fwd_inputs_vec;
  batched_fwd_inputs_vec.reserve(inputs.size());
  for (auto i = 0; i < inputs.size(); ++i) {
    proto::ForwardInput pb_fwd_input;
    forward_input_to_proto(inputs[i], &pb_fwd_input);
    batched_fwd_inputs_vec.push_back(std::move(pb_fwd_input));
  }
  ADD_VECTOR_TO_PROTO(pb_batched_fwd_inputs.mutable_micro_inputs(),
                      batched_fwd_inputs_vec);
  // call ExecuteModel with callback
  auto done = new ExecuteModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteModel(
      &done->cntl, &pb_batched_fwd_inputs, &done->pb_output, done);
  return true;
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

void ExecuteModelClosure::Run() {
  std::unique_ptr<ExecuteModelClosure> self_guard(this);

  if (cntl.Failed()) {
    LOG(ERROR) << "Execute_model_async failed. Error code : "
               << cntl.ErrorCode() << ", error message : " << cntl.ErrorText();
  }

  RawForwardOutput raw_forward_output;
  proto_to_forward_output(pb_output, raw_forward_output);
  promise.setValue(raw_forward_output);

  return;
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
}  // namespace xllm