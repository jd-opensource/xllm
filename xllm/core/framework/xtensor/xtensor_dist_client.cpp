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

#include "xtensor_dist_client.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>
#include <thread>

#include "common/global_flags.h"

namespace xllm {

XTensorDistClient::XTensorDistClient(int32_t global_rank,
                                     const std::string& server_address,
                                     const torch::Device& device)
    : global_rank_(global_rank), device_(device) {
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;
  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc channel to " << server_address;
    return;
  }
  stub_.reset(new proto::XTensorDist_Stub(&channel_));
  wait_for_server_ready(server_address);
}

bool XTensorDistClient::wait_for_server_ready(
    const std::string& server_address) {
  proto::Status req;
  proto::Status resp;

  int try_count = 0;
  brpc::Controller cntl;
  const int sleep_time_second = 3;
  while (try_count < FLAGS_max_reconnect_count) {
    cntl.Reset();
    stub_->Hello(&cntl, &req, &resp, nullptr);
    if (cntl.Failed() || !resp.ok()) {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    } else {
      LOG(INFO) << "XTensorDistClient connected to server: " << server_address
                << ", global_rank: " << global_rank_;
      break;
    }
    try_count++;
  }
  if (try_count >= FLAGS_max_reconnect_count) {
    LOG(ERROR) << "XTensorDistClient Hello failed, global_rank: "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }
  return true;
}

folly::SemiFuture<MemoryInfo> XTensorDistClient::get_memory_info_async() {
  folly::Promise<MemoryInfo> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Status req;
    proto::MemoryInfoResponse resp;
    brpc::Controller cntl;
    stub_->GetMemoryInfo(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "GetMemoryInfo failed: " << cntl.ErrorText();
      promise.setValue(MemoryInfo{0, 0});
      return;
    }
    // Returns 0 for both fields on failure
    promise.setValue(MemoryInfo{resp.available_memory(), resp.total_memory()});
  });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::init_phy_page_pool_async(
    int64_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, num_pages, promise = std::move(promise)]() mutable {
        proto::InitPhyPagePoolRequest req;
        req.set_num_pages(num_pages);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->InitPhyPagePool(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "InitPhyPagePool failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::map_to_kv_tensors_async(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_id,
                        offsets = offsets,
                        promise = std::move(promise)]() mutable {
    proto::KvTensorRequest req;
    req.set_model_id(model_id);
    for (offset_t offset : offsets) {
      req.add_offsets(offset);
    }
    proto::Status resp;
    brpc::Controller cntl;
    stub_->MapToKvTensors(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "MapToKvTensors failed: " << cntl.ErrorText();
      promise.setValue(false);
      return;
    }
    promise.setValue(resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::unmap_from_kv_tensors_async(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_id,
                        offsets = offsets,
                        promise = std::move(promise)]() mutable {
    proto::KvTensorRequest req;
    req.set_model_id(model_id);
    for (offset_t offset : offsets) {
      req.add_offsets(offset);
    }
    proto::Status resp;
    brpc::Controller cntl;
    stub_->UnmapFromKvTensors(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "UnmapFromKvTensors failed: " << cntl.ErrorText();
      promise.setValue(false);
      return;
    }
    promise.setValue(resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::create_weight_tensor_async(
    const std::string& model_id,
    int64_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_id, num_pages, promise = std::move(promise)]() mutable {
        proto::WeightTensorRequest req;
        req.set_model_id(model_id);
        req.set_num_pages(num_pages);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->CreateWeightTensor(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "CreateWeightTensor failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::map_weight_tensor_async(
    const std::string& model_id,
    int64_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_id, num_pages, promise = std::move(promise)]() mutable {
        proto::WeightTensorRequest req;
        req.set_model_id(model_id);
        req.set_num_pages(num_pages);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->MapWeightTensor(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "MapWeightTensor failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::unmap_weight_tensor_async(
    const std::string& model_id) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_id, promise = std::move(promise)]() mutable {
        proto::ModelRequest req;
        req.set_model_id(model_id);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->UnmapWeightTensor(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "UnmapWeightTensor failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

}  // namespace xllm
