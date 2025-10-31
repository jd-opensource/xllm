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

#include "rec_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include "absl/time/time.h"
#include "models/model_registry.h"
#include "runtime/rec_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "util/scope_guard.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {

RecMaster::RecMaster(const Options& options)
    : Master(options, EngineType::REC) {
  // Initialize with Rec engine type
  // The rest of the initialization follows the same pattern as LLMMaster
  CHECK(engine_->init());

  model_args_ = engine_->model_args();

  bool enable_decode_response_to_service = false;
  if (options_.enable_service_routing()) {
    XServiceClient* xservice_client = XServiceClient::get_instance();
    if (!xservice_client->init(options_.etcd_addr().value_or(""),
                               options_.xservice_addr().value_or(""),
                               options_.instance_name().value_or(""),
                               engine_->block_manager_pool())) {
      LOG(FATAL) << "XServiceClient init fail!";
      return;
    }
    auto service_config = xservice_client->get_config();
    enable_decode_response_to_service =
        service_config.enable_decode_response_to_service;
  }

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .dp_size(options_.dp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .enable_decode_response_to_service(enable_decode_response_to_service);
  scheduler_ = create_fixsteps_scheduler(engine_.get(), scheduler_options);

  // OmniRec model does not have a tokenizer
  chat_template_ = nullptr;
  tokenizer_ = nullptr;
  threadpool_ =
      std::make_unique<ThreadPool>(options_.num_request_handling_threads());
}

void RecMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "RecMaster is already running.";
    return;
  }
  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(5);
    while (!stopped_.load(std::memory_order_relaxed)) {
      // move scheduler forward
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });

  // Engine run method is not available, remove this call
}

RecMaster::~RecMaster() {
  // set stop flag
  stopped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void RecMaster::handle_request(std::string prompt,
                               std::optional<std::vector<int>> prompt_tokens,
                               std::optional<MMData> mm_data,
                               RequestParams sp,
                               OutputCallback callback) {
  // add one pending request
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  // add into the queue
  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         prompt_tokens = std::move(prompt_tokens),
                         mm_data = std::move(mm_data),
                         sp = std::move(sp),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(std::move(prompt),
                                    std::move(prompt_tokens),
                                    std::move(mm_data),
                                    sp,
                                    callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

std::shared_ptr<Request> RecMaster::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<MMData> mm_data,
    RequestParams sp,
    OutputCallback callback) {
  // For Rec model, prompt is expected to be empty and prompt_tokens should
  // contain the actual data Skip prompt empty check as mentioned in
  // requirements

  Timer timer;
  std::vector<int> local_prompt_tokens;

  if (prompt_tokens.has_value()) {
    local_prompt_tokens = std::move(prompt_tokens.value());
    LOG(INFO)
        << "[Rec DEBUG] generate_request - received prompt_tokens.size(): "
        << local_prompt_tokens.size()
        << ", prompt.length(): " << prompt.length();
  } else if (!mm_data.has_value()) {
    // sparse LLM
    LOG(ERROR) << "Rec model requires prompt_tokens/embedding to be provided";
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Rec model requires prompt_tokens/embedding to be provided");
    return nullptr;
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  int32_t max_context_len = model_args_.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }
  if (local_prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << local_prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  size_t capacity = local_prompt_tokens.size() + max_tokens +
                    options_.num_speculative_tokens() + /*bonus_token*/ 1;
  if (options_.enable_schedule_overlap()) {
    capacity += options_.num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  sampling_param.logprobs = sp.logprobs;
  sampling_param.top_logprobs = sp.top_logprobs;
  sampling_param.is_embeddings = sp.is_embeddings;
  sampling_param.beam_width = sp.beam_width;
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  // sampling_param.do_sample = sp.do_sample;

  bool stream = sp.streaming;
  // results cannot be streamed when best_of != n
  if (best_of != sp.n) {
    stream = false;
  }
  // std::unordered_set<int32_t> stop_tokens;
  // std::vector<std::vector<int32_t>> stop_sequences;
  // StoppingChecker stopping_checker(
  //     max_tokens,
  //     max_context_len - options_.num_speculative_tokens(),
  //     ,
  //     model_args_.eos_token_id(),
  //     sp.ignore_eos,
  //     std::move(stop_tokens),
  //     std::move(stop_sequences));
  StoppingChecker stopping_checker;
  RequestState req_state(std::move(prompt),
                         std::move(local_prompt_tokens),
                         mm_data.value_or(MMData{}),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_.enable_schedule_overlap(),
                         callback,
                         nullptr,
                         sp.decode_address);
  req_state.is_rec_model = true;
  req_state.bos_token_id = model_args_.bos_token_id();
  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id);
  return request;
}

}  // namespace xllm