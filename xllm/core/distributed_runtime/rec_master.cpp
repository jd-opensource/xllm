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

#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <unordered_set>

#include "common/macros.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/request/mm_data.h"
#include "models/model_registry.h"
#include "rec_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "util/scope_guard.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {

namespace {

constexpr int32_t kDefaultPlaceholderToken = 20152019;

RecType get_rec_type(const ModelArgs& model_args) {
  const auto& model_type = model_args.model_type();
  if (model_type == "onerec") {
    return RecType::kOneRec;
  }
  if (model_type == "qwen2" || model_type == "qwen3") {
    return RecType::kLlmRec;
  }
  return RecType::kNone;
}

bool process_onerec_inputs(
    const std::optional<std::vector<int>>& prompt_tokens,
    const std::optional<std::vector<proto::InferInputTensor>>& input_tensors,
    std::vector<int32_t>* local_prompt_tokens,
    MMData* processed_mm_data,
    OutputCallback callback) {
  if (prompt_tokens.has_value() && input_tensors.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "prompt_tokens and input_tensors cannot both be set");
    return false;
  }

  if (prompt_tokens.has_value()) {
    local_prompt_tokens->assign(prompt_tokens.value().begin(),
                                prompt_tokens.value().end());
  }

  if (input_tensors.has_value()) {
    MMDict mm_dict;
    mm_dict.reserve(input_tensors->size());
    for (const auto& tensor : input_tensors.value()) {
      mm_dict[tensor.name()] =
          util::convert_rec_tensor_to_torch(tensor).to(torch::kBFloat16);
    }
    *processed_mm_data = MMData(MMType::EMBEDDING, mm_dict);
  }

  if (local_prompt_tokens->empty() && !processed_mm_data->valid()) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Rec model requires prompt_tokens or input_tensors to be provided");
    return false;
  }

  return true;
}

bool process_llmrec_raw_inputs(
    std::optional<std::vector<int>> input_tokens,
    std::optional<std::vector<int>> input_indices,
    std::optional<std::vector<std::vector<float>>> input_embedding,
    const ModelArgs& model_args,
    std::vector<int32_t>* local_prompt_tokens,
    MMData* processed_mm_data,
    OutputCallback callback) {
  std::vector<int32_t> local_input_tokens;
  std::vector<int32_t> local_input_indices;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;
  torch::Tensor input_embedding_tensor;
  int64_t embedding_rows = 0;

  if (input_tokens.has_value()) {
    const auto& tokens = input_tokens.value();
    local_input_tokens.reserve(tokens.size());
    for (const auto token : tokens) {
      local_input_tokens.push_back(static_cast<int32_t>(token));
    }
    if (!local_input_tokens.empty()) {
      input_tokens_tensor =
          torch::from_blob(local_input_tokens.data(),
                           {static_cast<int64_t>(local_input_tokens.size())},
                           torch::dtype(torch::kInt32).device(torch::kCPU))
              .clone();
      processed_mm_data->add(
          MMType::EMBEDDING, LLM_REC_INPUT_TOKENS, input_tokens_tensor);
      local_prompt_tokens->assign(local_input_tokens.begin(),
                                  local_input_tokens.end());
    }
  }

  if (input_indices.has_value()) {
    if (!input_tokens.has_value()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "LLMRec input indices require input tokens");
      return false;
    }
    const auto& indices = input_indices.value();
    local_input_indices.reserve(indices.size());
    for (const auto index : indices) {
      local_input_indices.push_back(static_cast<int32_t>(index));
    }
    if (local_input_indices.size() != local_input_tokens.size()) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "LLMRec input indices size does not match input tokens");
      return false;
    }
    if (!local_input_indices.empty()) {
      input_indices_tensor =
          torch::from_blob(local_input_indices.data(),
                           {static_cast<int64_t>(local_input_indices.size())},
                           torch::dtype(torch::kInt32).device(torch::kCPU))
              .clone();
      processed_mm_data->add(
          MMType::EMBEDDING, LLM_REC_INPUT_INDICES, input_indices_tensor);
    }
  }

  if (input_embedding.has_value()) {
    const auto& embedding_vec = input_embedding.value();
    if (embedding_vec.empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "LLMRec input embedding is empty");
      return false;
    }
    const int64_t rows = static_cast<int64_t>(embedding_vec.size());
    const int64_t cols = static_cast<int64_t>(embedding_vec[0].size());
    if (cols != model_args.hidden_size()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "LLMRec input embedding has invalid hidden size");
      return false;
    }

    std::vector<float> flat_data;
    flat_data.reserve(static_cast<size_t>(rows * cols));
    for (const auto& row : embedding_vec) {
      flat_data.insert(flat_data.end(), row.begin(), row.end());
    }
    input_embedding_tensor =
        torch::from_blob(flat_data.data(),
                         {rows, cols},
                         torch::dtype(torch::kFloat32).device(torch::kCPU))
            .clone();
    processed_mm_data->add(
        MMType::EMBEDDING, LLM_REC_INPUT_EMBEDDING, input_embedding_tensor);
    embedding_rows = rows;
    local_prompt_tokens->insert(local_prompt_tokens->end(),
                                static_cast<size_t>(embedding_rows),
                                kDefaultPlaceholderToken);
  }

  if (!local_input_indices.empty()) {
    const int64_t total_size =
        static_cast<int64_t>(local_input_tokens.size()) + embedding_rows;
    std::unordered_set<int32_t> seen;
    seen.reserve(local_input_indices.size());
    for (const auto index : local_input_indices) {
      if (index < 0 || index >= total_size) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "LLMRec input indices contain invalid values");
        return false;
      }
      if (!seen.insert(index).second) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "LLMRec input indices contain duplicate values");
        return false;
      }
    }
  }

  if (local_prompt_tokens->empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is empty");
    return false;
  }

  return true;
}

}  // namespace

RecMaster::RecMaster(const Options& options)
    : Master(options, EngineType::REC) {
  // Initialize with Rec engine type
  // The rest of the initialization follows the same pattern as LLMMaster
  CHECK(engine_->init());

  model_args_ = engine_->model_args();
  rec_type_ = get_rec_type(model_args_);
  if (rec_type_ == RecType::kNone) {
    LOG(ERROR) << "Unsupported rec model_type: " << model_args_.model_type();
  }

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
  scheduler_ = create_fixed_steps_scheduler(engine_.get(), scheduler_options);

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

void RecMaster::handle_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    RequestParams sp,
    OutputCallback callback) {
  if (rec_type_ != RecType::kOneRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "OneRec should use onerec input interface");
    return;
  }
  schedule_request(std::move(sp),
                   std::move(callback),
                   [this,
                    prompt = std::move(prompt),
                    prompt_tokens = std::move(prompt_tokens),
                    input_tensors = std::move(input_tensors)](
                       const RequestParams& params, OutputCallback cb) mutable {
                     return generate_request(std::move(prompt),
                                             std::move(prompt_tokens),
                                             std::move(input_tensors),
                                             params,
                                             std::move(cb));
                   });
}

void RecMaster::handle_request(
    std::optional<std::vector<int>> input_tokens,
    std::optional<std::vector<int>> input_indices,
    std::optional<std::vector<std::vector<float>>> input_embedding,
    RequestParams sp,
    OutputCallback callback) {
  if (rec_type_ != RecType::kLlmRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "LLMRec should use raw input interface");
    return;
  }
  schedule_request(std::move(sp),
                   std::move(callback),
                   [this,
                    input_tokens = std::move(input_tokens),
                    input_indices = std::move(input_indices),
                    input_embedding = std::move(input_embedding)](
                       const RequestParams& params, OutputCallback cb) mutable {
                     return generate_request(std::move(input_tokens),
                                             std::move(input_indices),
                                             std::move(input_embedding),
                                             params,
                                             std::move(cb));
                   });
}

void RecMaster::schedule_request(RequestParams sp,
                                 OutputCallback callback,
                                 RequestBuilder build_request) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  threadpool_->schedule([this,
                         sp = std::move(sp),
                         callback = std::move(cb),
                         build_request = std::move(build_request)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = build_request(sp, std::move(callback));
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
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback) {
  // For Rec model, prompt is expected to be empty and prompt_tokens should
  // contain the actual data Skip prompt empty check as mentioned in
  // requirements

  if (rec_type_ == RecType::kNone) {
    LOG(ERROR) << "Unsupported rec model_type: " << model_args_.model_type();
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        std::string("Unsupported rec model_type: ") + model_args_.model_type());
    return nullptr;
  }

  Timer timer;
  std::vector<int32_t> local_prompt_tokens;
  MMData processed_mm_data;
  bool processed_ok = false;

  if (rec_type_ == RecType::kOneRec) {
    processed_ok = process_onerec_inputs(prompt_tokens,
                                         input_tensors,
                                         &local_prompt_tokens,
                                         &processed_mm_data,
                                         callback);
  }

  if (!processed_ok) {
    return nullptr;
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  return build_request_common(std::move(prompt),
                              std::move(local_prompt_tokens),
                              std::move(processed_mm_data),
                              sp,
                              callback,
                              rec_type_ == RecType::kLlmRec);
}

std::shared_ptr<Request> RecMaster::generate_request(
    std::optional<std::vector<int>> input_tokens,
    std::optional<std::vector<int>> input_indices,
    std::optional<std::vector<std::vector<float>>> input_embedding,
    const RequestParams& sp,
    OutputCallback callback) {
  if (rec_type_ != RecType::kLlmRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "LLMRec inputs require rec_type kLlmRec");
    return nullptr;
  }

  Timer timer;
  std::vector<int32_t> local_prompt_tokens;
  MMData processed_mm_data;
  if (!process_llmrec_raw_inputs(std::move(input_tokens),
                                 std::move(input_indices),
                                 std::move(input_embedding),
                                 model_args_,
                                 &local_prompt_tokens,
                                 &processed_mm_data,
                                 callback)) {
    return nullptr;
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  return build_request_common(std::string(""),
                              std::move(local_prompt_tokens),
                              std::move(processed_mm_data),
                              sp,
                              callback,
                              true);
}

std::shared_ptr<Request> RecMaster::build_request_common(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    OutputCallback callback,
    bool build_stop_checker) {
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

  size_t capacity = prompt_tokens.size() + max_tokens +
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
    sampling_param.logprobs = true;
  }

  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  StoppingChecker stopping_checker;
  if (build_stop_checker) {
    std::unordered_set<int32_t> stop_tokens;
    if (sp.stop_token_ids.has_value()) {
      const auto& stop_token_ids = sp.stop_token_ids.value();
      stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
    } else {
      stop_tokens = model_args_.stop_token_ids();
    }

    std::vector<std::vector<int32_t>> stop_sequences;
    if (sp.stop.has_value()) {
      if (!tokenizer_) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Tokenizer is required for stop sequences");
        return nullptr;
      }
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

    stopping_checker =
        StoppingChecker(max_tokens,
                        max_context_len - options_.num_speculative_tokens(),
                        model_args_.eos_token_id(),
                        sp.ignore_eos,
                        std::move(stop_tokens),
                        std::move(stop_sequences));
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
                         options_.enable_schedule_overlap(),
                         callback,
                         nullptr,
                         sp.decode_address);
  req_state.rec_type = rec_type_;
  req_state.bos_token_id = model_args_.bos_token_id();
  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id);
  return request;
}

}  // namespace xllm
