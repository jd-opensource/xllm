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

#include "vmm_submitter.h"

#include <chrono>
#include <thread>

#include <glog/logging.h>

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
    VMMRequest *req = const_cast<VMMRequest *>(work_queue_.try_peek());
    if (req == nullptr) {
        return false;
    }
    if (has_conflict(req->va)) {
        return false;
    }
    if (req->op_type == OpType::UNMAP) {
        defer_request(*req);
        work_queue_.dequeue();
        return false;
    }
    execute_map(*req);
    work_queue_.dequeue();
    return true;
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

void VMMWorker::schedule(int max_ops) {
    int ops_done = 0;
    
    while (ops_done < max_ops) {
        if (step_current() || step_deferred()) {
            ops_done++;
        } else {
            break;
        }
    }
}

bool VMMWorker::has_conflict(VirtPtr va) {
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

void VMMWorker::notify_completion(VMMSubmitter* submitter, RequestId request_id,
                                  OpType op_type, bool success) {
    if (!submitter) {
        return;
    }
    
    VMMCompletion completion(request_id, op_type, success);
    
    if (!submitter->push_completion(completion)) {
        LOG(WARNING) << "Failed to push completion for request " << request_id;
    }
}

VMMManager::~VMMManager() {
    shutdown();
}

bool VMMManager::init_device(int32_t device_id) {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    
    if (workers_.find(device_id) != workers_.end()) {
        LOG(WARNING) << "Device " << device_id << " already initialized";
        return false;
    }

    // Create worker using private factory method
    auto worker = create_worker(device_id);
    if (!worker) {
        LOG(ERROR) << "Failed to create worker for device " << device_id;
        return false;
    }

    worker->start();
    workers_[device_id] = worker;
    LOG(INFO) << "Initialized worker for device " << device_id;
    return true;
}

void VMMManager::shutdown() {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    if (shutdown_flag_.exchange(true)) {
        return;  // Already shutting down or done
    }

    LOG(INFO) << "Shutting down VMMManager...";

    for (auto& [device_id, worker] : workers_) {
        worker->stop();
    }

    workers_.clear();
    LOG(INFO) << "VMMManager shutdown complete";
}

std::unique_ptr<VMMSubmitter> VMMManager::create_submitter(int32_t device_id) {
    // Use private constructor to create submitter
    return std::unique_ptr<VMMSubmitter>(new VMMSubmitter(device_id));
}

std::shared_ptr<VMMWorker> VMMManager::get_worker(int32_t device_id) {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    
    auto it = workers_.find(device_id);
    if (it == workers_.end()) {
        return nullptr;
    }
    
    return it->second;
}

std::shared_ptr<VMMWorker> VMMManager::create_worker(int32_t device_id) {
    // Use private constructor to create worker
    return std::shared_ptr<VMMWorker>(new VMMWorker(device_id));
}

}  // namespace vmm
}  // namespace xllm