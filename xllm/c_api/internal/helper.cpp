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

#include "helper.h"

#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <string>

#include "core/util/uuid.h"

namespace xllm {
namespace helper {
namespace {
thread_local ShortUUID short_uuid;
static std::atomic<bool> g_glog_inited = false;
static pthread_mutex_t g_log_init_mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

std::string generate_request_id() {
  return "xllm-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

void init_log(const std::string& log_dir) {
  if (g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (!g_glog_inited.load(std::memory_order_relaxed)) {
    google::InitGoogleLogging("xllm");

    std::string log_prefix = log_dir.empty() ? "./" : log_dir + "/";
    google::SetLogDestination(google::INFO,
                              (log_prefix + "xllm.log.INFO.").c_str());
    google::SetLogDestination(google::WARNING,
                              (log_prefix + "xllm.log.WARNING.").c_str());
    google::SetLogDestination(google::ERROR,
                              (log_prefix + "xllm.log.ERROR.").c_str());
    google::SetStderrLogging(google::FATAL);
    g_glog_inited.store(true, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void shutdown_log() {
  if (!g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (g_glog_inited.load(std::memory_order_relaxed)) {
    google::ShutdownGoogleLogging();
    g_glog_inited.store(false, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void set_init_options(BackendType backend_type,
                      const XLLM_InitOptions* init_options,
                      XLLM_InitOptions* xllm_init_options) {
  if (backend_type == BackendType::LLM) {
    memcpy(xllm_init_options,
           &XLLM_INIT_LLM_OPTIONS_DEFAULT,
           sizeof(XLLM_InitOptions));
  } else if (backend_type == BackendType::REC) {
    memcpy(xllm_init_options,
           &XLLM_INIT_REC_OPTIONS_DEFAULT,
           sizeof(XLLM_InitOptions));
  }

  if (NULL == init_options) {
    return;
  }

  xllm_init_options->enable_mla = init_options->enable_mla;
  xllm_init_options->enable_chunked_prefill =
      init_options->enable_chunked_prefill;
  xllm_init_options->enable_prefix_cache = init_options->enable_prefix_cache;
  xllm_init_options->enable_disagg_pd = init_options->enable_disagg_pd;
  xllm_init_options->enable_pd_ooc = init_options->enable_pd_ooc;
  xllm_init_options->enable_schedule_overlap =
      init_options->enable_schedule_overlap;
  xllm_init_options->enable_shm = init_options->enable_shm;

  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, transfer_listen_port);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, nnodes);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, node_rank);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, dp_size);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, ep_size);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, block_size);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, max_cache_size);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, max_tokens_per_batch);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, max_seqs_per_batch);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, max_tokens_per_chunk_for_prefill);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, num_speculative_tokens);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, num_request_handling_threads);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, expert_parallel_degree);
  XLLM_SET_FIELD_IF_NONZERO(xllm_init_options, init_options, server_idx);
  XLLM_SET_FIELD_IF_NONZERO(
      xllm_init_options, init_options, max_memory_utilization);

  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, task);
  XLLM_SET_FIELD_IF_NONEMPTY(
      xllm_init_options, init_options, communication_backend);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, instance_role);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, device_ip);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, master_node_addr);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, xservice_addr);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, instance_name);
  XLLM_SET_FIELD_IF_NONEMPTY(
      xllm_init_options, init_options, kv_cache_transfer_mode);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, log_dir);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, draft_model);
  XLLM_SET_FIELD_IF_NONEMPTY(xllm_init_options, init_options, draft_devices);

  return;
}

void transfer_request_params(InferenceType inference_type,
                             const XLLM_RequestParams* request_params,
                             xllm::RequestParams* xllm_request_params) {
  XLLM_RequestParams final_request_params;
  if (inference_type == InferenceType::LLM_COMPLETIONS ||
      inference_type == InferenceType::LLM_CHAT_COMPLETIONS) {
    memcpy(&final_request_params,
           &XLLM_LLM_REQUEST_PARAMS_DEFAULT,
           sizeof(XLLM_RequestParams));
  } else if (inference_type == InferenceType::REC_COMPLETIONS ||
             inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
    memcpy(&final_request_params,
           &XLLM_REC_REQUEST_PARAMS_DEFAULT,
           sizeof(XLLM_RequestParams));
  }

  if (nullptr != request_params) {
    final_request_params.echo = request_params->echo;
    final_request_params.offline = request_params->offline;
    final_request_params.logprobs = request_params->logprobs;
    final_request_params.ignore_eos = request_params->ignore_eos;

    XLLM_SET_FIELD_IF_NONZERO(&final_request_params, request_params, n);
    XLLM_SET_FIELD_IF_NONZERO(&final_request_params, request_params, best_of);
    XLLM_SET_FIELD_IF_NONZERO(&final_request_params, request_params, slo_ms);
    XLLM_SET_FIELD_IF_NONZERO(&final_request_params, request_params, top_k);
    XLLM_SET_FIELD_IF_NONZERO(&final_request_params, request_params, top_p);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, max_tokens);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, frequency_penalty);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, presence_penalty);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, repetition_penalty);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, beam_width);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, top_logprobs);
    XLLM_SET_FIELD_IF_NONZERO(
        &final_request_params, request_params, temperature);

    XLLM_SET_FIELD_IF_NONEMPTY(
        &final_request_params, request_params, request_id);
  }

  xllm_request_params->echo = final_request_params.echo;
  xllm_request_params->offline = final_request_params.offline;
  xllm_request_params->logprobs = final_request_params.logprobs;
  xllm_request_params->ignore_eos = final_request_params.ignore_eos;

  xllm_request_params->best_of = final_request_params.best_of;
  xllm_request_params->slo_ms = final_request_params.slo_ms;
  xllm_request_params->top_k = final_request_params.top_k;
  xllm_request_params->top_p = final_request_params.top_p;
  xllm_request_params->n = final_request_params.n;
  xllm_request_params->max_tokens = final_request_params.max_tokens;
  xllm_request_params->frequency_penalty =
      final_request_params.frequency_penalty;
  xllm_request_params->presence_penalty = final_request_params.presence_penalty;
  xllm_request_params->repetition_penalty =
      final_request_params.repetition_penalty;
  xllm_request_params->beam_width = final_request_params.beam_width;
  xllm_request_params->top_logprobs = final_request_params.top_logprobs;
  xllm_request_params->temperature = final_request_params.temperature;
  xllm_request_params->request_id = final_request_params.request_id;

  return;
}

XLLM_Response* build_error_response(const std::string& request_id,
                                    XLLM_StatusCode status_code,
                                    const std::string& error_info) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = status_code;
  strncpy(
      response->error_info, error_info.c_str(), XLLM_ERROR_INFO_MAX_LEN - 1);
  response->error_info[XLLM_ERROR_INFO_MAX_LEN - 1] = '\0';

  XLLM_SET_META_STRING_FIELD(response->id, request_id);

  LOG(ERROR) << "Request [" << request_id << "] error: " << error_info
             << " (code: " << static_cast<int>(response->status_code) << ")";

  return response;
}

XLLM_Response* build_success_response(const InferenceType& inference_type,
                                      const RequestOutput& output,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = XLLM_StatusCode::kSuccess;
  response->created = created_time;
  XLLM_SET_META_STRING_FIELD(response->id, request_id);
  XLLM_SET_META_STRING_FIELD(response->model, model);

  if (inference_type == InferenceType::LLM_COMPLETIONS ||
      inference_type == InferenceType::REC_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "text_completion");
  } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
             inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "chat.completion");
  }

  response->choices.entries_size = output.outputs.size();
  response->choices.entries = new XLLM_Choice[response->choices.entries_size]();
  CHECK(nullptr != response->choices.entries);

  for (int i = 0; i < output.outputs.size(); i++) {
    const auto& seq_output = output.outputs[i];
    XLLM_Choice& choice = response->choices.entries[i];
    choice.index = seq_output.index;

    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::REC_COMPLETIONS) {
      size_t text_len = seq_output.text.length();
      choice.text = new char[text_len + 1];
      CHECK(nullptr != choice.text);
      strncpy(choice.text, seq_output.text.c_str(), text_len + 1);
      choice.text[text_len] = '\0';
    } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      choice.message = new XLLM_ChatMessage();
      CHECK(nullptr != choice.message);

      snprintf(choice.message->role, sizeof(choice.message->role), "assistant");
      size_t text_len = seq_output.text.length();
      choice.message->content = new char[text_len + 1];
      CHECK(nullptr != choice.message->content);
      strncpy(choice.message->content, seq_output.text.c_str(), text_len + 1);
      choice.message->content[text_len] = '\0';
    }

    if (seq_output.finish_reason.has_value()) {
      XLLM_SET_META_STRING_FIELD(choice.finish_reason,
                                 seq_output.finish_reason.value());
    }

    if (seq_output.logprobs.has_value()) {
      choice.logprobs.entries_size = seq_output.logprobs.value().size();
      choice.logprobs.entries =
          new XLLM_LogProb[choice.logprobs.entries_size]();
      CHECK(nullptr != choice.logprobs.entries);
      for (int j = 0; j < seq_output.logprobs.value().size(); j++) {
        const auto& logprob = seq_output.logprobs.value()[j];
        XLLM_LogProb& xllm_logprob = choice.logprobs.entries[j];

        xllm_logprob.token_id = logprob.token_id;
        xllm_logprob.logprob = logprob.logprob;
      }
    }
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response->usage.prompt_tokens =
        static_cast<int32_t>(usage.num_prompt_tokens);
    response->usage.completion_tokens =
        static_cast<int32_t>(usage.num_generated_tokens);
    response->usage.total_tokens = static_cast<int32_t>(usage.num_total_tokens);
  }

  return response;
}

template <typename HandlerType, typename InputType>
XLLM_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  CHECK(nullptr != handler);

  std::string request_id;
  if (nullptr != request_params && strlen(request_params->request_id) > 0) {
    request_id = request_params->request_id;
  } else {
    request_id = generate_request_id();
  }

  if (!handler->initialized) {
    return build_error_response(
        request_id, XLLM_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  if (std::find(handler->model_ids.begin(),
                handler->model_ids.end(),
                model_id) == handler->model_ids.end()) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  xllm::RequestParams xllm_request_params;
  transfer_request_params(inference_type, request_params, &xllm_request_params);
  xllm_request_params.request_id = request_id;

  const int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<XLLM_Response*>>();
    auto future = promise_ptr->getSemiFuture();

    auto on_request_complete = [model_id,
                                request_id,
                                created_time,
                                inference_type,
                                weak_promise = std::weak_ptr(promise_ptr)](
                                   const RequestOutput& req_output) -> bool {
      if (auto locked_promise = weak_promise.lock()) {
        try {
          locked_promise->setValue(build_success_response(
              inference_type, req_output, request_id, created_time, model_id));
          return true;
        } catch (const std::exception& e) {
          LOG(ERROR) << "Build response failed: " << e.what();
          locked_promise->setValue(build_error_response(
              request_id,
              XLLM_StatusCode::kInternalError,
              "Build response failed: " + std::string(e.what())));
        }
      }
      return false;
    };

    if constexpr (std::is_same_v<HandlerType, XLLM_LLM_Handler>) {
      handler->master->handle_request(input,
                                      std::nullopt,
                                      xllm_request_params,
                                      std::nullopt,
                                      on_request_complete);
    } else if constexpr (std::is_same_v<HandlerType, XLLM_REC_Handler>) {
      if constexpr (std::is_same_v<InputType, std::vector<int>>) {
        handler->master->handle_request(
            "", input, std::nullopt, xllm_request_params, on_request_complete);
      } else {
        handler->master->handle_request(input,
                                        std::nullopt,
                                        std::nullopt,
                                        xllm_request_params,
                                        on_request_complete);
      }
    } else {
      CHECK(false);
    }

    return std::move(future)
        .via(handler->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([request_id](
                     folly::Try<XLLM_Response*>&& result) -> XLLM_Response* {
          if (result.hasValue()) return std::move(result).value();

          std::string error_msg;
          XLLM_StatusCode code = XLLM_StatusCode::kInternalError;
          try {
            result.throwUnlessValue();
          } catch (const folly::FutureTimeout& e) {
            error_msg = "Request timed out: " + std::string(e.what());
            code = XLLM_StatusCode::kTimeout;
          } catch (const std::exception& e) {
            error_msg = "Inference failed: " + std::string(e.what());
          } catch (...) {
            error_msg = "Inference failed with unknown exception";
          }
          return build_error_response(request_id, code, error_msg);
        })
        .get();

  } catch (...) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kInternalError,
                                "Critical error in inference pipeline");
  }
}

void xllm_free_response(XLLM_Response* resp) {
  if (nullptr == resp) {
    return;
  }

  if (nullptr != resp->choices.entries) {
    for (int i = 0; i < resp->choices.entries_size; ++i) {
      XLLM_Choice& choice = resp->choices.entries[i];

      if (nullptr != choice.text) {
        delete[] choice.text;
        choice.text = nullptr;
      }

      if (nullptr != choice.message) {
        if (nullptr != choice.message->content) {
          delete[] choice.message->content;
          choice.message->content = nullptr;
        }
        delete choice.message;
        choice.message = nullptr;
      }

      if (nullptr != choice.logprobs.entries) {
        delete[] choice.logprobs.entries;
        choice.logprobs.entries = nullptr;
      }
      choice.logprobs.entries_size = 0;
    }

    delete[] resp->choices.entries;
    resp->choices.entries = nullptr;
  }

  resp->choices.entries_size = 0;
  delete resp;

  return;
}

// 1. LLM Handler + const char* (text completions)
template XLLM_Response* handle_inference_request<XLLM_LLM_Handler, const char*>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 2. LLM Handler + std::vector<xllm::Message> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_LLM_Handler, std::vector<xllm::Message>>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 3. REC Handler + const char* (REC completions)
template XLLM_Response* handle_inference_request<XLLM_REC_Handler, const char*>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 4. REC Handler + std::vector<xllm::Message> (REC chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<xllm::Message>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 5. REC Handler + std::vector<int> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<int>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<int>& input,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);
}  // namespace helper
}  // namespace xllm