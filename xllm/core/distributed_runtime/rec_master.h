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
#include <functional>
#include <thread>

#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/rec_type.h"
#include "master.h"
#include "rec.pb.h"
#include "rec_engine.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"
#include "util/threadpool.h"

namespace xllm {

class RecMaster : public Master {
 public:
  explicit RecMaster(const Options& options);
  ~RecMaster();

  // handle a request, the engine will execute the request asynchronously
  // completion/encode
  void handle_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      OutputCallback callback);

  void handle_request(
      std::optional<std::vector<int>> input_tokens,
      std::optional<std::vector<int>> input_indices,
      std::optional<std::vector<std::vector<float>>> input_embedding,
      RequestParams sp,
      OutputCallback callback);

  // start the handling loop
  void run() override;

 private:
  using RequestBuilder =
      std::function<std::shared_ptr<Request>(const RequestParams&,
                                             OutputCallback)>;

  void schedule_request(RequestParams sp,
                        OutputCallback callback,
                        RequestBuilder build_request);

  std::shared_ptr<Request> generate_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      const RequestParams& sp,
      OutputCallback callback);

  std::shared_ptr<Request> generate_request(
      std::optional<std::vector<int>> input_tokens,
      std::optional<std::vector<int>> input_indices,
      std::optional<std::vector<std::vector<float>>> input_embedding,
      const RequestParams& sp,
      OutputCallback callback);

  std::shared_ptr<Request> build_request_common(
      std::string prompt,
      std::vector<int32_t> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      OutputCallback callback,
      bool build_stop_checker);

  std::unique_ptr<FixedStepsScheduler> scheduler_;
  // model args
  ModelArgs model_args_;
  RecType rec_type_ = RecType::kNone;
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
