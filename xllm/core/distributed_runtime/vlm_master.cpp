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

#include "vlm_master.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/mm_data.h"
#include "framework/request/request.h"
#include "models/model_registry.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#include "speculative_engine.h"
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "vlm_engine.h"

namespace xllm {

VLMMaster::VLMMaster(const Options& options)
    : Master(options, EngineType::VLM) {
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
  scheduler_options.max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .enable_decode_response_to_service(enable_decode_response_to_service)
      .disable_ttft_profiling(options_.disable_ttft_profiling())
      .enable_forward_interruption(options_.enable_forward_interruption())
      // TODO: support later for VLM.
      .enable_schedule_overlap(false)
      .server_idx(options_.server_idx());
  scheduler_ = create_continuous_scheduler(engine_.get(), scheduler_options);

  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }

  // construct chat template
  chat_template_ =
      std::make_unique<JinjaChatTemplate>(engine_->tokenizer_args());

  // create input processor
  auto input_processor_factory =
      ModelRegistry::get_input_processor_factory(model_args_.model_type());
  if (input_processor_factory == nullptr) {
    LOG(ERROR) << "No input processor defined for model type: "
               << model_args_.model_type();
  } else {
    input_processor_ = input_processor_factory(model_args_);
  }

  // create image processor
  auto image_processor_factory =
      ModelRegistry::get_image_processor_factory(model_args_.model_type());
  if (image_processor_factory == nullptr) {
    LOG(ERROR) << "No image processor defined for model type: "
               << model_args_.model_type();
  } else {
    image_processor_ = image_processor_factory(model_args_);
  }

  // construct tokenizer and handling threads
  tokenizer_ = engine_->tokenizer()->clone();
  threadpool_ =
      std::make_unique<ThreadPool>(options_.num_request_handling_threads());
}

VLMMaster::~VLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void VLMMaster::handle_request(const std::string& prompt,
                               MMData& mm_data,
                               const RequestParams& sp,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         prompt = std::move(prompt),
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

    auto request = generate_request(prompt, mm_data, sp, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_request(const std::vector<Message>& messages,
                               const RequestParams& sp,
                               const std::string& payload,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         messages = std::move(messages),
                         sp = std::move(sp),
                         payload = std::move(payload),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_chat);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(messages, sp, payload, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_batch_request(const std::vector<std::string>& prompts,
                                     std::vector<MMData>& mm_datas,
                                     const std::vector<RequestParams>& sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   mm_datas[i],
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void VLMMaster::handle_batch_request(
    const std::vector<std::vector<Message>>& conversations,
    const std::vector<RequestParams>& sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  const size_t num_requests = conversations.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::string(),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void VLMMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "VLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    running_.store(true, std::memory_order_relaxed);
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void VLMMaster::generate() {
  DCHECK(options_.enable_schedule_overlap())
      << "Mode generate does not support schedule overlap yet.";
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

std::shared_ptr<Request> VLMMaster::generate_request(std::string& prompt,
                                                     MMData& mm_data,
                                                     const RequestParams& sp,
                                                     OutputCallback callback) {
  if (prompt.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is empty");
    return nullptr;
  }
  Timer timer;
  input_processor_->process(prompt, mm_data);

  std::vector<int> prompt_tokens;
  if (!tokenizer_->encode(prompt, &prompt_tokens)) {
    LOG(ERROR) << "Failed to encode prompt: " << prompt;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to encode prompt");
    return nullptr;
  }
  input_processor_->find_mm_spans(prompt_tokens, mm_data);

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  // TODO: prompt_token is not enough, need to add image token size
  int32_t max_context_len = model_args_.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }
  if (prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens, TODO: add image token size as well.
  const size_t capacity = prompt_tokens.size() + max_tokens + 1;
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
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  // sampling_param.do_sample = sp.do_sample;

  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else {
    stop_tokens = model_args_.stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  StoppingChecker stopping_checker(max_tokens,
                                   max_context_len,
                                   model_args_.eos_token_id(),
                                   sp.ignore_eos,
                                   std::move(stop_tokens),
                                   std::move(stop_sequences));

  // results cannot be streamed when best_of != n
  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         false, /*enable_schedule_overlap*/
                         callback,
                         nullptr);
  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id);

  // add one sequence, rest will be added by scheduler
  return request;
}

std::shared_ptr<Request> VLMMaster::generate_request(
    const std::vector<Message>& messages,
    const RequestParams& sp,
    const std::string& payload,
    OutputCallback callback) {
  Timer timer;
  static MMInputTransfer mm_input_transfer;
  MMInput mm_inputs(payload);
  if (!mm_input_transfer.trans(messages, mm_inputs)) {
    LOG(ERROR) << "mm input trans failed.";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "MM input transfer trans failed.");
    return nullptr;
  }

  MMData mm_data;
  if (!mm_inputs.empty() && !image_processor_->process(mm_inputs, mm_data)) {
    LOG(ERROR) << " image processor process failed.";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Image processor process failed.");
    return nullptr;
  }

  auto prompt = chat_template_->apply(messages);
  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages");
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  return generate_request(prompt.value(), mm_data, sp, callback);
}

}  // namespace xllm
