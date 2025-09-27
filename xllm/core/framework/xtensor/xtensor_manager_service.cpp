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

#include "xtensor_manager_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <vector>

namespace xllm {

XTensorManagerService::XTensorManagerService(int32_t global_rank,
                                             int32_t world_size,
                                             const torch::Device& d)
    : global_rank_(global_rank),
      world_size_(world_size),
      device_(d),
      initialized_(false) {}

void XTensorManagerService::set_xtensor_manager(
    std::unique_ptr<XTensorManager> xtensor_manager) {
  xtensor_manager_ = std::move(xtensor_manager);
  initialized_ = true;
}

void XTensorManagerService::Hello(::google::protobuf::RpcController* controller,
                                  const proto::Status* request,
                                  proto::Status* response,
                                  ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
  return;
}

void XTensorManagerService::Allocate(
    ::google::protobuf::RpcController* controller,
    const proto::AllocatePagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    int32_t seq_id = request->seq_id();
    size_t num_tokens = request->num_tokens();
    auto future = xtensor_manager_->allocate_async(seq_id, num_tokens);
    bool status = std::move(future).get();
    response->set_ok(status);
  });
  return;
}

void XTensorManagerService::Deallocate(
    ::google::protobuf::RpcController* controller,
    const proto::SeqId* request,
    proto::Empty* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    xtensor_manager_->deallocate_async(request->seq_id());
  });
  return;
}

void XTensorManagerService::NumFreePagesPerLayer(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::NumPages* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    response->set_num_pages(xtensor_manager_->num_free_pages_per_layer());
  });
  return;
}

void XTensorManagerService::NumUsedPagesPerLayer(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::NumPages* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    response->set_num_pages(xtensor_manager_->num_used_pages_per_layer());
  });
  return;
}

void XTensorManagerService::KvCacheUtilization(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::Utilization* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    response->set_utilization(xtensor_manager_->kv_cache_utilization());
  });
  return;
}

}  // namespace xllm