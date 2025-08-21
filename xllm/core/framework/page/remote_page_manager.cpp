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

#include "remote_page_manager.h"

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
#include "page_manager.pb.h"

namespace xllm {
RemotePageManager::RemotePageManager(int32_t global_rank,
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
  stub_.reset(new proto::DistributePageManager_Stub(&channel_));

  wait_for_server_ready(server_address);
}

bool RemotePageManager::wait_for_server_ready(
    const std::string& server_address) {
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
      LOG(INFO) << "RemotePageManager Hello connected, server_address: "
                << server_address << ", global_rank_: " << global_rank_;
      break;
    }

    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "RemotePageManager Hello method failed, global_rank_ is "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }

  return true;
}

bool RemotePageManager::allocate(int32_t& seq_id, size_t num_tokens) {
  proto::AllocatePagesRequest req;
  req.set_seq_id(seq_id);
  req.set_num_tokens(num_tokens);
  proto::Status resp;
  brpc::Controller cntl;
  stub_->Allocate(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Allocate method failed: " << cntl.ErrorText();
    return false;
  }
  return resp.ok();
}

void RemotePageManager::deallocate(int32_t seq_id) {
  proto::SeqId req;
  req.set_seq_id(seq_id);
  proto::Empty resp;
  brpc::Controller cntl;
  stub_->Deallocate(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Deallocate method failed: " << cntl.ErrorText();
  }
}

void RemotePageManager::cache(int32_t seq_id) {
  proto::SeqId req;
  req.set_seq_id(seq_id);
  proto::Empty resp;
  brpc::Controller cntl;
  stub_->Cache(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Cache method failed: " << cntl.ErrorText();
  }
}

folly::SemiFuture<bool> RemotePageManager::allocate_async(int32_t& seq_id,
                                                          size_t num_tokens) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule(
      [this, seq_id, num_tokens, promise = std::move(promise)]() mutable {
        proto::AllocatePagesRequest req;
        req.set_seq_id(seq_id);
        req.set_num_tokens(num_tokens);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->Allocate(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "Allocate method failed: " << cntl.ErrorText();
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<folly::Unit> RemotePageManager::deallocate_async(
    int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    proto::SeqId req;
    req.set_seq_id(seq_id);
    proto::Empty resp;
    brpc::Controller cntl;
    stub_->Deallocate(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Deallocate method failed: " << cntl.ErrorText();
    }
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<folly::Unit> RemotePageManager::cache_async(int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    proto::SeqId req;
    req.set_seq_id(seq_id);
    proto::Empty resp;
    brpc::Controller cntl;
    stub_->Cache(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Cache method failed: " << cntl.ErrorText();
    }
    promise.setValue();
  });
  return future;
}

size_t RemotePageManager::num_free_pages_per_layer() const {
  proto::Empty req;
  proto::NumPages resp;
  brpc::Controller cntl;
  stub_->NumFreePagesPerLayer(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "NumFreePagesPerLayer method failed: " << cntl.ErrorText();
  }
  return resp.num_pages();
}

size_t RemotePageManager::num_used_pages_per_layer() const {
  proto::Empty req;
  proto::NumPages resp;
  brpc::Controller cntl;
  stub_->NumUsedPagesPerLayer(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "NumUsedPagesPerLayer method failed: " << cntl.ErrorText();
  }
  return resp.num_pages();
}

double RemotePageManager::kv_cache_utilization() const {
  proto::Empty req;
  proto::Utilization resp;
  brpc::Controller cntl;
  stub_->KvCacheUtilization(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "KvCacheUtilization method failed: " << cntl.ErrorText();
  }
  return resp.utilization();
}

folly::SemiFuture<size_t> RemotePageManager::num_free_pages_per_layer_async() {
  folly::Promise<size_t> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::NumPages resp;
    brpc::Controller cntl;
    stub_->NumFreePagesPerLayer(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "NumFreePagesPerLayer method failed: " << cntl.ErrorText();
    }
    promise.setValue(resp.num_pages());
  });
  return future;
}

folly::SemiFuture<size_t> RemotePageManager::num_used_pages_per_layer_async() {
  folly::Promise<size_t> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::NumPages resp;
    brpc::Controller cntl;
    stub_->NumUsedPagesPerLayer(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "NumUsedPagesPerLayer method failed: " << cntl.ErrorText();
    }
    promise.setValue(resp.num_pages());
  });
  return future;
}

}  // namespace xllm