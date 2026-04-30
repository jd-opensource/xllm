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

#include "speech_service_impl.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "core/common/global_flags.h"
#include "core/framework/request/mm_data.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"

namespace xllm {
namespace {

constexpr const char* kRuntimeUnavailableMessage =
    "/v1/audio/speech decoder is not integrated in this build";

std::string to_lower_ascii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return std::tolower(ch);
      });
  return value;
}

bool is_blank(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") == std::string::npos;
}

bool is_supported_response_format(const std::string& format) {
  return format == "wav" || format == "pcm" || format == "flac" ||
         format == "mp3" || format == "aac" || format == "opus";
}

Status prepare_speech_request(proto::SpeechRequest* request,
                              const absl::flat_hash_set<std::string>& models,
                              const std::string& default_model) {
  if (request == nullptr) {
    return Status(StatusCode::INVALID_ARGUMENT, "speech request is null");
  }
  if (is_blank(request->input())) {
    return Status(StatusCode::INVALID_ARGUMENT, "input cannot be empty");
  }

  if (!request->has_model() || request->model().empty()) {
    request->set_model(default_model);
  }
  if (!models.contains(request->model())) {
    return Status(StatusCode::UNKNOWN, "Model not supported");
  }

  std::string response_format = request->has_response_format()
                                    ? to_lower_ascii(request->response_format())
                                    : "wav";
  if (response_format.empty()) {
    response_format = "wav";
  }
  if (!is_supported_response_format(response_format)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "response_format must be one of wav, pcm, flac, mp3, aac, "
                  "opus");
  }
  request->set_response_format(response_format);

  if (!request->has_speed()) {
    request->set_speed(1.0);
  }
  if (!request->has_stream_format() || request->stream_format().empty()) {
    request->set_stream_format("audio");
  }
  if (!request->has_stream()) {
    request->set_stream(false);
  }
  if (!request->has_task_type() || request->task_type().empty()) {
    const bool is_base =
        request->has_ref_audio() || request->speaker_embedding_size() > 0;
    request->set_task_type(is_base ? "Base" : "CustomVoice");
  }
  if (!request->has_language() || request->language().empty()) {
    request->set_language("Auto");
  }
  if (!request->has_max_new_tokens() || request->max_new_tokens() <= 0) {
    request->set_max_new_tokens(2048);
  }

  return Status();
}

}  // namespace

SpeechServiceImpl::SpeechServiceImpl(VLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models),
      master_(master),
      default_model_(models.empty() ? FLAGS_model_id : models.front()) {
  CHECK(master_ != nullptr);
}

void SpeechServiceImpl::process_async_impl(std::shared_ptr<SpeechCall> call) {
  proto::SpeechRequest request = call->request();
  auto status = prepare_speech_request(&request, models_, default_model_);
  if (!status.ok()) {
    call->finish_with_error(status.code(), status.message());
    return;
  }

  RequestParams request_params(
      request, call->get_x_request_id(), call->get_x_request_time());

  auto prompt = request.input();
  if (request.has_instructions() && !is_blank(request.instructions())) {
    prompt = request.instructions() + "\n" + request.input();
  }

  master_->handle_request(
      std::move(prompt),
      MMData(),
      std::move(request_params),
      [call](const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }

        if (req_output.finished || req_output.cancelled) {
          return call->finish_with_error(StatusCode::UNAVAILABLE,
                                         kRuntimeUnavailableMessage);
        }
        return true;
      });
}

}  // namespace xllm
