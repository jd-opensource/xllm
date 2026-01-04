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

#include "xtensor_dist_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <vector>

#include "common/device_monitor.h"
#include "phy_page_pool.h"
#include "platform/device.h"
#include "xtensor_allocator.h"

#if defined(USE_NPU)
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#elif defined(USE_MLU)
#include <torch_mlu/csrc/framework/core/caching_allocator.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace xllm {

XTensorDistService::XTensorDistService(int32_t global_rank,
                                       int32_t world_size,
                                       const torch::Device& device)
    : global_rank_(global_rank),
      world_size_(world_size),
      device_(device),
      initialized_(false) {}

void XTensorDistService::Hello(::google::protobuf::RpcController* controller,
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
}

void XTensorDistService::GetMemoryInfo(
    ::google::protobuf::RpcController* controller,
    const proto::Status* request,
    proto::MemoryInfoResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    Device device(device_);
    device.set_device();

    // Empty torch cache to get accurate memory info
    size_t torch_cache = 0;
    size_t torch_largest_block = 0;
    int32_t device_id = device_.index();

#if defined(USE_NPU)
    c10_npu::NPUCachingAllocator::emptyCache();
    c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(device_id);
    c10_npu::NPUCachingAllocator::cacheInfo(
        device_id, &torch_cache, &torch_largest_block);
#elif defined(USE_MLU)
    torch_mlu::MLUCachingAllocator::emptyCache();
#elif defined(USE_CUDA) || defined(USE_ILU)
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    // Update device monitor
    DeviceMonitor::get_instance().set_total_memory(device_id, total_memory);

    LOG(INFO) << "GetMemoryInfo: global_rank=" << global_rank_
              << ", available_memory=" << available_memory + torch_cache
              << ", total_memory=" << total_memory;

    // Returns 0 for both fields on failure (handled by caller)
    response->set_available_memory(available_memory + torch_cache);
    response->set_total_memory(total_memory);
  });
}

void XTensorDistService::InitPhyPagePool(
    ::google::protobuf::RpcController* controller,
    const proto::InitPhyPagePoolRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    int64_t num_pages = request->num_pages();
    LOG(INFO) << "InitPhyPagePool: global_rank=" << global_rank_
              << ", num_pages=" << num_pages;

    try {
      // Initialize PhyPagePool with specified number of pages
      PhyPagePool::get_instance().init(device_, num_pages);
      response->set_ok(true);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to init PhyPagePool: " << e.what();
      response->set_ok(false);
    }
  });
}

void XTensorDistService::MapToKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call XTensorAllocator to map
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.map_to_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void XTensorDistService::UnmapFromKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call XTensorAllocator to unmap
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.unmap_from_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void XTensorDistService::CreateWeightTensor(
    ::google::protobuf::RpcController* controller,
    const proto::WeightTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    int64_t num_pages = request->num_pages();
    LOG(INFO) << "CreateWeightTensor: global_rank=" << global_rank_
              << ", model_id=" << model_id << ", num_pages=" << num_pages;

    // Call XTensorAllocator to create weight tensor (no mapping)
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.create_weight_tensor(model_id, num_pages);
    response->set_ok(success);
  });
}

void XTensorDistService::MapWeightTensor(
    ::google::protobuf::RpcController* controller,
    const proto::WeightTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    int64_t num_pages = request->num_pages();
    LOG(INFO) << "MapWeightTensor: global_rank=" << global_rank_
              << ", model_id=" << model_id << ", num_pages=" << num_pages;

    // Call XTensorAllocator to map weight tensor
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.map_weight_tensor(model_id, num_pages);
    response->set_ok(success);
  });
}

void XTensorDistService::UnmapWeightTensor(
    ::google::protobuf::RpcController* controller,
    const proto::ModelRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    LOG(INFO) << "UnmapWeightTensor: global_rank=" << global_rank_
              << ", model_id=" << model_id;

    // Call XTensorAllocator to unmap weight tensor
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.unmap_weight_tensor(model_id);
    response->set_ok(success);
  });
}

}  // namespace xllm
