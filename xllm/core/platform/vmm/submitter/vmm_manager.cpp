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

#include "vmm_manager.h"
#include "vmm_worker.h"
#include "vmm_submitter.h"

namespace xllm {
namespace vmm {

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