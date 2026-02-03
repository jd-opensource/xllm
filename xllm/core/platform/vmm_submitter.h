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

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <folly/concurrency/UnboundedQueue.h>

#include "vmm_api.h"

namespace xllm {
namespace vmm {

enum class OpType {
    MAP,
    UNMAP
};

class VMMManager;
class VMMWorker;
class VMMSubmitter;

using RequestId = uint64_t;

struct VMMRequest {
    OpType op_type;
    VirtPtr va;
    PhyMemHandle phy;
    size_t size;
    RequestId request_id;
    VMMSubmitter* submitter;
    
    VMMRequest() 
        : op_type(OpType::MAP), va(0), phy(0), size(0), request_id(0), submitter(nullptr) {}
    
    VMMRequest(OpType type, VirtPtr v, PhyMemHandle p, size_t s, RequestId id, VMMSubmitter* sub)
        : op_type(type), va(v), phy(p), size(s), request_id(id), submitter(sub) {}
};

struct VMMCompletion {
    RequestId request_id;
    OpType op_type;
    bool success;
    
    VMMCompletion() : request_id(0), op_type(OpType::MAP), success(false) {}
    
    VMMCompletion(RequestId id, OpType type, bool succ)
        : request_id(id), op_type(type), success(succ) {}
};

using RequestQueue = folly::UMPSCQueue<VMMRequest, /* Mayblock */ false>;
using CompletionQueue = folly::USPSCQueue<VMMCompletion, /* Mayblock */ false>;

// VMMSubmitter: Client interface for submitting requests
// Can only be constructed by VMMManager
class VMMSubmitter {
public:
    ~VMMSubmitter();
    
    VMMSubmitter(const VMMSubmitter&) = delete;
    VMMSubmitter& operator=(const VMMSubmitter&) = delete;
    VMMSubmitter(VMMSubmitter&& other) = delete;
    VMMSubmitter& operator=(VMMSubmitter&& other) = delete;
    
    RequestId map(VirtPtr va, PhyMemHandle phy);
    
    RequestId unmap(VirtPtr va, size_t aligned_size);
    
    size_t poll_completions(size_t max_completions = 32);
    
    bool all_map_done() const;
    
    bool all_unmap_done() const;
    
    void wait_all();
    
    bool is_connected() const { return connected_ && worker_ != nullptr; }

private:
    explicit VMMSubmitter(int32_t device_id);

    bool connect(int32_t device_id);

    void disconnect();
    
    bool push_completion(const VMMCompletion& completion);
    
    int32_t device_id_;

    std::shared_ptr<VMMWorker> worker_;

    bool connected_;
    
    CompletionQueue completion_queue_;
    
    RequestId next_request_id_;

    uint64_t pending_map_{0};
    uint64_t pending_unmap_{0};

    friend class VMMManager;
    friend class VMMWorker;
};

// VMMWorker: Worker thread that executes VMM operations
// Can only be constructed by VMMManager
class VMMWorker {
public:
    ~VMMWorker();
    
    VMMWorker(const VMMWorker&) = delete;
    VMMWorker& operator=(const VMMWorker&) = delete;
    VMMWorker(VMMWorker&&) = delete;
    VMMWorker& operator=(VMMWorker&&) = delete;
    
    void start();
    
    void stop();
    
    bool submit_request(const VMMRequest& req);
    
private:
    explicit VMMWorker(int32_t device_id);
    
    void worker_loop();
    
    bool step_current();
    
    bool step_deferred();

    void defer_request(const VMMRequest& req);
    
    void schedule(int max_ops);
    
    bool has_conflict(VirtPtr va);
    
    void execute_map(VMMRequest& req);
    
    void execute_unmap(VMMRequest& req);
    
    void notify_completion(VMMSubmitter* submitter, RequestId request_id, 
                          OpType op_type, bool success);

private:
    int32_t device_id_;
    std::unique_ptr<std::thread> worker_thread_;
    std::atomic<bool> running_;
    
    RequestQueue work_queue_;
    
    std::unordered_set<VirtPtr> deferred_va_;
    
    std::deque<VMMRequest> deferred_requests_;
    
    // Only VMMManager can construct VMMWorker
    friend class VMMManager;
};

class VMMManager {
public:
    static VMMManager& get_instance() {
        static VMMManager instance;
        return instance;
    }
    
    bool init_device(int32_t device_id);
    
    void shutdown();
    
    std::unique_ptr<VMMSubmitter> create_submitter(int32_t device_id);
    
    std::shared_ptr<VMMWorker> get_worker(int32_t device_id);

private:
    VMMManager() = default;
    ~VMMManager();
    
    VMMManager(const VMMManager&) = delete;
    VMMManager& operator=(const VMMManager&) = delete;
    VMMManager(VMMManager&&) = delete;
    VMMManager& operator=(VMMManager&&) = delete;
        
    std::shared_ptr<VMMWorker> create_worker(int32_t device_id);
    
    std::unordered_map<int32_t, std::shared_ptr<VMMWorker>> workers_;
    mutable std::mutex workers_mutex_;
    std::atomic<bool> shutdown_flag_{false};
    
    friend class VMMSubmitter;
};

}  // namespace vmm
}  // namespace xllm