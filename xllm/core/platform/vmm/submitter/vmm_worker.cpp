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

#include <glog/logging.h>

#include "core/common/macros.h"
#include "vmm_worker.h"
#include "vmm_submitter.h"

namespace xllm {
namespace vmm {

VMMWorker::VMMWorker(int32_t device_id)
    : device_id_(device_id),
      running_(false) {
}

VMMWorker::~VMMWorker() {
    stop();
}

void VMMWorker::start() {
    if (running_.load()) {
        LOG(WARNING) << "Worker for device " << device_id_ << " already started";
        return;
    }
    
    running_.store(true);
    worker_thread_ = std::make_unique<std::thread>(&VMMWorker::worker_loop, this);
}

void VMMWorker::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }
}

bool VMMWorker::submit_request(const VMMRequest& req) {
    work_queue_.enqueue(req);
    return true;
}

void VMMWorker::worker_loop() {
    LOG(INFO) << "Worker for device " << device_id_ << " started";

    while (running_.load()) {
        schedule(32);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

bool VMMWorker::step_current() {
    VMMRequest *req = nullptr;
    while (nullptr != (req = const_cast<VMMRequest *>(work_queue_.try_peek()))) {
        if (has_conflict(req->va)) {
            return false;
        }
        if (req->op_type == OpType::UNMAP) {
            defer_request(*req);
            work_queue_.dequeue();
            continue;
        }
        execute_map(*req);
        work_queue_.dequeue();
        return true;
    }
    return false;
}

bool VMMWorker::step_deferred() {
    if (deferred_requests_.empty()) {
        return false;
    }
    auto req = deferred_requests_.front();
    if (req.op_type == OpType::MAP) {
        execute_map(req);
    } else {
        execute_unmap(req);
    }
    deferred_va_.erase(req.va);
    deferred_requests_.pop_front();
    return true;
}

void VMMWorker::schedule(int32_t max_ops) {
    int ops_done = 0;
    
    while (ops_done < max_ops) {
        if (step_current() || step_deferred()) {
            ops_done++;
        } else {
            break;
        }
    }
}

bool VMMWorker::has_conflict(VirPtr va) {
    return deferred_va_.find(va) != deferred_va_.end();
}

void VMMWorker::execute_map(VMMRequest& req) {
    vmm::map(req.va, req.phy);
    notify_completion(req.submitter, req.request_id, OpType::MAP, true);
}

void VMMWorker::execute_unmap(VMMRequest& req) {
    vmm::unmap(req.va, req.size);
    notify_completion(req.submitter, req.request_id, OpType::UNMAP, true);
}

void VMMWorker::defer_request(const VMMRequest& req) {
    deferred_va_.insert(req.va);
    deferred_requests_.push_back(req);
}

void VMMWorker::notify_completion(VMMSubmitter* submitter, uint64_t request_id,
                                  OpType op_type, bool success) {
    if (!submitter) {
        return;
    }
    
    VMMCompletion completion(request_id, op_type, success);
    
    if (!submitter->push_completion(completion)) {
        LOG(WARNING) << "Failed to push completion for request " << request_id;
    }
}

}  // namespace vmm
}  // namespace xllm