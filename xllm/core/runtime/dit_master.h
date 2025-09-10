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

#include <folly/Function.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "runtime/master.h"
// #include "runtime/mm_engine.h"
#include "scheduler/continuous_scheduler.h"
#include "util/concurrent_queue.h"

namespace xllm {

class Call;

class DITMaster : public Master {
 public:
  explicit DITMaster(const Options& options);
  ~DITMaster();

  // handle a request, the engine will execute the request asynchronously
  void handle_request(ImageRequestParams sp,
                      std::optional<Call*> call,
                      ImageOutputCallback callback);

  // batch generation
  void handle_batch_request(std::vector<ImageRequestParams> sp,
                            BatchImageOutputCallback callback);

  // start running loop
  void run() override;

  // generate will run all request done at once,
  // this is a blocking call
  void generate();

  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<std::string>& addrs,
                      std::vector<int64_t>& k_cache_ids,
                      std::vector<int64_t>& v_cache_ids);

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<std::string>& device_ips,
                    const std::vector<uint16_t>& ports,
                    const int32_t dp_size);

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<std::string>& device_ips,
                      const std::vector<uint16_t>& ports,
                      const int32_t dp_size);

 private:
  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace xllm