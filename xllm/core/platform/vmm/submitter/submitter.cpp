/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "submitter.h"

#include <chrono>
#include <thread>

#include <glog/logging.h>

#include "manager.h"
#include "worker.h"

namespace xllm {
namespace vmm {

VMMSubmitter::VMMSubmitter(int device_id)
    : device_id_(device_id),
      worker_(nullptr),
      connected_(false),
      next_request_id_(1),
      pending_map_(0),
      pending_unmap_(0) {

    connect(device_id);
}

VMMSubmitter::~VMMSubmitter() {
    wait_all();
    disconnect();
}

bool VMMSubmitter::connect(int32_t device_id) {
    if (connected_) {
        LOG(WARNING) << "Already connected to device " << device_id_;
        return false;
    }

    worker_ = VMMManager::get_instance().get_worker(device_id_);
    if (!worker_) {
        LOG(ERROR) << "Failed to get worker for device " << device_id
                   << ". Device not initialized?";
        return false;
    }

    connected_ = true;
    LOG(INFO) << "Submitter connected to device " << device_id;
    return true;
}

void VMMSubmitter::disconnect() {
    if (connected_) {
        LOG(INFO) << "Disconnecting submitter from device " << device_id_;
        wait_all();
        worker_.reset();
        connected_ = false;
    }
}

RequestId VMMSubmitter::map(VirtPtr va, PhyMemHandle phy) {
    if (!is_connected()) {
        LOG(ERROR) << "Not connected or worker destroyed";
        return 0;
    }
    
    RequestId request_id = next_request_id_++;
    VMMRequest req(OpType::MAP, va, phy, 0, request_id, this);
    
    if (!worker_->submit_request(req)) {
        LOG(ERROR) << "Failed to submit map request";
        return 0;
    }

    pending_map_++;
    return request_id;
}

RequestId VMMSubmitter::unmap(VirtPtr va, size_t aligned_size) {
    if (!is_connected()) {
        LOG(ERROR) << "Not connected or worker destroyed";
        return 0;
    }
    
    RequestId request_id = next_request_id_++;
    VMMRequest req(OpType::UNMAP, va, 0, aligned_size, request_id, this);
    
    if (!worker_->submit_request(req)) {
        LOG(ERROR) << "Failed to submit unmap request";
        return 0;
    }

    pending_unmap_++;
    return request_id;
}

size_t VMMSubmitter::poll_completions(size_t max_completions) {
    size_t count = 0;
    VMMCompletion completion;

    while (count < max_completions && completion_queue_.try_dequeue(completion)) {
        if (completion.op_type == OpType::MAP) {
            if (pending_map_ > 0) pending_map_--;
        } else {
            if (pending_unmap_ > 0) pending_unmap_--;
        }
        if (!completion.success) {
            LOG(ERROR) << "Operation failed: request_id=" << completion.request_id
                       << ", type=" << (completion.op_type == OpType::MAP ? "MAP" : "UNMAP");
        }
        count++;
    }

    return count;
}

bool VMMSubmitter::all_map_done() const {
    return pending_map_ == 0;
}

bool VMMSubmitter::all_unmap_done() const {
    return pending_unmap_ == 0;
}

void VMMSubmitter::wait_all() {
    while (!all_map_done() || !all_unmap_done()) {
        poll_completions(32);
        std::this_thread::yield();
    }
}

bool VMMSubmitter::push_completion(const VMMCompletion& completion) {
    completion_queue_.enqueue(completion);
    return true;
}

}  // namespace vmm
}  // namespace xllm