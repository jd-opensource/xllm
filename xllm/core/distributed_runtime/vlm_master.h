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
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "common/options.h"
#include "common/types.h"
#include "engine.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/request/mm_input.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "master.h"
#include "scheduler/continuous_scheduler.h"
#include "xllm/processors/input_processor.h"

namespace xllm {

struct MMData;
class ImageProcessor;

class VLMMaster : public Master {
 public:
  explicit VLMMaster(const Options& options);
  ~VLMMaster();

  // completion
  void handle_request(const std::string& prompt,
                      const MMData& mm_data,
                      const RequestParams& sp,
                      OutputCallback callback);

  // chat
  void handle_request(const std::vector<Message>& messages,
                      const MMData& mm_data,
                      const RequestParams& sp,
                      OutputCallback callback);

  // chat
  void handle_request(const std::vector<Message>& messages,
                      const RequestParams& sp,
                      const std::string& payload,
                      OutputCallback callback);

  // batch completion
  void handle_batch_request(const std::vector<std::string>& prompts,
                            const std::vector<MMData>& mm_datas,
                            const std::vector<RequestParams>& sps,
                            BatchOutputCallback callback);

  // batch chat
  void handle_batch_request(
      const std::vector<std::vector<Message>>& conversations,
      const std::vector<MMData>& mm_datas,
      const std::vector<RequestParams>& sps,
      BatchOutputCallback callback);

  // start the handling loop
  void run() override;

  // generate will run all requests, this is an blocking call
  void generate();

  int get_image_limit() { return options_.limit_image_per_prompt(); }

 private:
  using Task = folly::Function<void()>;
  std::shared_ptr<Request> generate_request(std::string prompt,
                                            const MMData& mm_data,
                                            const RequestParams& sp,
                                            OutputCallback callback);

  std::shared_ptr<Request> generate_request(

      const std::vector<Message>& messages,
      const MMData& mm_data,
      const RequestParams& sp,
      OutputCallback callback);

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

  // input processor for vlm
  std::unique_ptr<InputProcessor> input_processor_;

  std::unique_ptr<ImageProcessor> image_processor_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace xllm
