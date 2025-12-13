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
#include "llm_engine.h"
#include "master.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {

class Call;

class LLMMaster : public Master {
 public:
  explicit LLMMaster(const Options& options);
  ~LLMMaster();

  // handle a request, the engine will execute the request asynchronously
  // completion/encode
  void handle_request(std::string prompt,
                      std::optional<std::vector<int>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  // chat
  void handle_request(std::vector<Message> messages,
                      std::optional<std::vector<int>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  // batch completion
  void handle_batch_request(std::vector<std::string> prompts,
                            std::vector<RequestParams> sp,
                            BatchOutputCallback callback);

  // batch chat
  void handle_batch_request(std::vector<std::vector<Message>> conversations,
                            std::vector<RequestParams> sp,
                            BatchOutputCallback callback);

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
  std::shared_ptr<Request> generate_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback);

  std::shared_ptr<Request> generate_request(
      const std::vector<Message>& messages,
      std::optional<std::vector<int>> prompt_tokens,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback);

 private:
  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  // we don't know if tokenizer is thread safe, so we create one for each thread
  // for now
  std::unique_ptr<Tokenizer> tokenizer_;

  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};

  std::string task_type_;
};

class LLMAssistantMaster : public Master {
 public:
  LLMAssistantMaster(const Options& options);
  ~LLMAssistantMaster() = default;
  void run() override;

  static void handle_signal(int signum) { running_ = false; }

 private:
  static volatile bool running_;
};

}  // namespace xllm
