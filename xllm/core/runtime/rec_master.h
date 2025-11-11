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
#include <thread>

#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "runtime/master.h"
#include "runtime/rec_engine.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/fixsteps_scheduler.h"
#include "util/threadpool.h"

namespace xllm {

class RecMaster : public Master {
 public:
  explicit RecMaster(const Options& options);
  ~RecMaster();

  // handle a request, the engine will execute the request asynchronously
  // completion/encode
  void handle_request(std::string prompt,
                      std::optional<std::vector<int>> prompt_tokens,
                      std::optional<MMData> mm_data,
                      RequestParams sp,
                      OutputCallback callback);

  // start the handling loop
  void run() override;

 private:
  std::shared_ptr<Request> generate_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<MMData> mm_data,
      RequestParams sp,
      OutputCallback callback);

  std::unique_ptr<FixStepsScheduler> scheduler_;
  // model args
  ModelArgs model_args_;
  std::unique_ptr<ThreadPool> threadpool_;
  std::unique_ptr<Tokenizer> tokenizer_;
  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;
  // thread for moving forward the scheduler
  std::thread loop_thread_;
  // flag to stop the loop
  std::atomic<bool> stopped_{false};

  // flag to indicate if the handler is running
  std::atomic<bool> running_{false};
};

}  // namespace xllm