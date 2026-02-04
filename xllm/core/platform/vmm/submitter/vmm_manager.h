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

#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>

#include "core/common/macros.h"

namespace xllm {
namespace vmm {

class VMMSubmitter;
class VMMWorker;

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
        
        DISALLOW_COPY_AND_MOVE(VMMManager);
            
        std::shared_ptr<VMMWorker> create_worker(int32_t device_id);
        
        std::unordered_map<int32_t, std::shared_ptr<VMMWorker>> workers_;
        mutable std::mutex workers_mutex_;
        std::atomic<bool> shutdown_flag_{false};
};

}  // namespace vmm
}  // namespace xllm