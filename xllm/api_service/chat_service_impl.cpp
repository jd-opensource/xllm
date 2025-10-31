/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "chat_service_impl.h"

#include <absl/strings/escaping.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>

#include "core/common/instance_name.h"
#include "core/common/types.h"
#include "core/framework/request/mm_input_helper.h"
#include "core/framework/request/request_params.h"
#include "core/runtime/llm_master.h"
#include "core/runtime/vlm_master.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "function_call/function_call.h"

namespace xllm {
namespace {

struct ToolCallResult {
  std::optional<google::protobuf::RepeatedPtrField<proto::ToolCall>> tool_calls;
  std::string text;
  std::string finish_reason;
};

ToolCallResult process_tool_calls(std::string text,
                                  const std::vector<xllm::JsonTool>& tools,
                                  const std::string& parser_format,
                                  std::string finish_reason,
                                  google::protobuf::Arena* arena = nullptr) {
  ToolCallResult result;

  function_call::FunctionCallParser parser(tools, parser_format);

  if (!parser.has_tool_call(text)) {
    result.text = std::move(text);
    result.finish_reason = std::move(finish_reason);
    return result;
  }

  if (finish_reason == "stop") {
    result.finish_reason = "tool_calls";
  } else {
    result.finish_reason = std::move(finish_reason);
  }

  try {
    auto [parsed_text, call_info_list] = parser.parse_non_stream(text);
    result.text = std::move(parsed_text);

    google::protobuf::RepeatedPtrField<proto::ToolCall> tool_calls;

    for (const auto& call_info : call_info_list) {
      proto::ToolCall* tool_call =
          arena ? google::protobuf::Arena::CreateMessage<proto::ToolCall>(arena)
                : new proto::ToolCall();

      tool_call->set_id(function_call::utils::generate_tool_call_id());
      tool_call->set_type("function");

      auto* function = tool_call->mutable_function();
      if (call_info.name) {
        function->set_name(*call_info.name);
      }
      function->set_arguments(call_info.parameters);

      tool_calls.AddAllocated(tool_call);
    }

    result.tool_calls = std::move(tool_calls);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Tool call parsing error: " << e.what();
  }

  return result;
}

void set_logprobs(proto::ChatChoice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  proto_logprobs->mutable_content()->Reserve(logprobs.value().size());
  for (const auto& logprob : logprobs.value()) {
    auto* logprob_proto = proto_logprobs->add_content();
    logprob_proto->set_token(logprob.token);
    logprob_proto->set_token_id(logprob.token_id);
    logprob_proto->set_logprob(logprob.logprob);

    if (logprob.top_logprobs.has_value()) {
      for (const auto& top_logprob : logprob.top_logprobs.value()) {
        auto* top_logprob_proto = logprob_proto->add_top_logprobs();
        top_logprob_proto->set_token(top_logprob.token);
        top_logprob_proto->set_token_id(top_logprob.token_id);
        top_logprob_proto->set_logprob(top_logprob.logprob);
      }
    }
  }
}

struct StreamingState {
  std::vector<function_call::JsonTool> tools;
  std::string parser_format;

  std::vector<std::unique_ptr<function_call::FunctionCallParser>> parsers;
  std::vector<bool> has_tool_calls;

  StreamingState(const std::vector<function_call::JsonTool>& tools,
                 const std::string& parser_format)
      : tools(tools), parser_format(parser_format) {
    if (!tools.empty() && !parser_format.empty()) {
      parsers.resize(1);
      has_tool_calls.resize(1, false);
      parsers[0] = std::make_unique<function_call::FunctionCallParser>(
          tools, parser_format);
    }
  }

  function_call::FunctionCallParser* get_parser_for_sequence(size_t index) {
    if (tools.empty() || parser_format.empty()) {
      return nullptr;
    }

    if (index >= parsers.size()) {
      parsers.resize(index + 1);
      has_tool_calls.resize(index + 1, false);
    }

    if (!parsers[index]) {
      parsers[index] = std::make_unique<function_call::FunctionCallParser>(
          tools, parser_format);
    }

    return parsers[index].get();
  }
};

template <typename ChatCall>
bool send_tool_call_chunk(std::shared_ptr<ChatCall> call,
                          size_t index,
                          const std::string& tool_call_id,
                          const std::string& function_name,
                          const std::string& arguments,
                          int tool_index,
                          const std::string& request_id,
                          int64_t created_time,
                          const std::string& model) {
  auto& response = call->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();

  auto* tool_call = delta->add_tool_calls();
  if (!tool_call_id.empty()) {
    tool_call->set_id(tool_call_id);
  }
  tool_call->set_index(tool_index);
  tool_call->set_type("function");

  auto* function = tool_call->mutable_function();
  if (!function_name.empty()) {
    function->set_name(function_name);
  }
  if (!arguments.empty()) {
    function->set_arguments(arguments);
  }

  return call->write(response);
}

template <typename ChatCall>
bool send_normal_text_chunk(std::shared_ptr<ChatCall> call,
                            size_t index,
                            const std::string& content,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model) {
  auto& response = call->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_content(content);

  return call->write(response);
}

template <typename ChatCall>
bool process_tool_call_stream(std::shared_ptr<ChatCall> call,
                              std::shared_ptr<StreamingState> streaming_state,
                              size_t index,
                              const std::string& delta,
                              const std::string& request_id,
                              int64_t created_time,
                              const std::string& model) {
  auto* parser = streaming_state->get_parser_for_sequence(index);
  if (!parser) {
    return true;
  }

  auto parse_result = parser->parse_streaming_increment(delta);

  if (!parse_result.normal_text.empty()) {
    if (!send_normal_text_chunk(call,
                                index,
                                parse_result.normal_text,
                                request_id,
                                created_time,
                                model)) {
      return false;
    }
  }

  for (const auto& call_item : parse_result.calls) {
    if (index >= streaming_state->has_tool_calls.size()) {
      streaming_state->has_tool_calls.resize(index + 1, false);
    }
    streaming_state->has_tool_calls[index] = true;

    std::string tool_call_id;
    std::string function_name;

    if (call_item.name.has_value()) {
      tool_call_id = function_call::utils::generate_tool_call_id();
      function_name = call_item.name.value();
    }

    if (!send_tool_call_chunk(call,
                              index,
                              tool_call_id,
                              function_name,
                              call_item.parameters,
                              call_item.tool_index,
                              request_id,
                              created_time,
                              model)) {
      return false;
    }
  }

  return true;
}

template <typename ChatCall>
bool check_for_unstreamed_tool_args(
    std::shared_ptr<ChatCall> call,
    std::shared_ptr<StreamingState> streaming_state,
    size_t index,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model) {
  auto* parser = streaming_state->get_parser_for_sequence(index);
  if (!parser) {
    return true;
  }

  auto* detector = parser->get_detector();
  if (!detector) {
    return true;
  }

  if (!detector->prev_tool_call_arr_.empty() &&
      !detector->streamed_args_for_tool_.empty()) {
    size_t tool_index = detector->prev_tool_call_arr_.size() - 1;
    if (tool_index < detector->streamed_args_for_tool_.size()) {
      const auto& expected_args = detector->prev_tool_call_arr_[tool_index];
      const std::string& actual_args =
          detector->streamed_args_for_tool_[tool_index];

      if (expected_args.find("arguments") != expected_args.end()) {
        const std::string& expected_call = expected_args.at("arguments");

        if (expected_call.length() > actual_args.length()) {
          std::string remaining_call =
              expected_call.substr(actual_args.length());

          if (!remaining_call.empty()) {
            return send_tool_call_chunk(call,
                                        index,
                                        "",
                                        "",
                                        remaining_call,
                                        static_cast<int>(tool_index),
                                        request_id,
                                        created_time,
                                        model);
          }
        }
      }
    }
  }

  return true;
}

template <typename ChatCall>
bool send_delta_to_client_brpc(
    std::shared_ptr<ChatCall> call,
    bool include_usage,
    std::unordered_set<size_t>* first_message_sent,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output,
    std::shared_ptr<StreamingState> streaming_state = nullptr) {
  auto& response = call->response();

  // send delta to client
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;

    if (first_message_sent->find(index) == first_message_sent->end()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      auto* message = choice->mutable_delta();
      message->set_role("assistant");
      message->set_content("");
      first_message_sent->insert(index);
      if (!call->write(response)) {
        return false;
      }
    }

    if (!seq_output.text.empty()) {
      if (streaming_state && streaming_state->get_parser_for_sequence(index)) {
        if (!process_tool_call_stream(call,
                                      streaming_state,
                                      index,
                                      seq_output.text,
                                      request_id,
                                      created_time,
                                      model)) {
          return false;
        }
      } else {
        response.Clear();
        response.set_object("chat.completion.chunk");
        response.set_id(request_id);
        response.set_created(created_time);
        response.set_model(model);
        auto* choice = response.add_choices();
        choice->set_index(index);
        set_logprobs(choice, seq_output.logprobs);
        auto* message = choice->mutable_delta();
        message->set_content(seq_output.text);
        if (!call->write(response)) {
          return false;
        }
      }
    }

    // Handle finish reason
    if (seq_output.finish_reason.has_value()) {
      // Check for unstreamed tool args before sending finish reason
      if (streaming_state && index < streaming_state->has_tool_calls.size() &&
          streaming_state->has_tool_calls[index]) {
        if (!check_for_unstreamed_tool_args(call,
                                            streaming_state,
                                            index,
                                            request_id,
                                            created_time,
                                            model)) {
          return false;
        }
      }

      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();

      if (streaming_state && index < streaming_state->has_tool_calls.size() &&
          streaming_state->has_tool_calls[index] &&
          seq_output.finish_reason.value() == "stop") {
        choice->set_finish_reason("tool_calls");
      } else {
        choice->set_finish_reason(std::move(seq_output.finish_reason.value()));
      }

      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value()) {
    response.Clear();
    const auto& usage = output.usage.value();
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
    if (!call->write(response)) {
      return false;
    }
  }

  if (output.finished || output.cancelled) {
    response.Clear();
    return call->finish();
  }
  return true;
}

template <typename ChatCall>
bool send_result_to_client_brpc(std::shared_ptr<ChatCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output,
                                const std::string& parser_format = "",
                                const std::vector<xllm::JsonTool>& tools = {}) {
  auto& response = call->response();
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    set_logprobs(choice, output.logprobs);
    auto* message = choice->mutable_message();
    message->set_role("assistant");

    auto set_output_and_finish_reason = [&]() {
      message->set_content(output.text);
      if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
    };

    if (!tools.empty() && !parser_format.empty()) {
      auto* arena = response.GetArena();
      auto result = process_tool_calls(output.text,
                                       tools,
                                       parser_format,
                                       output.finish_reason.value_or(""),
                                       arena);

      message->mutable_content()->swap(result.text);

      if (result.tool_calls) {
        auto& source_tool_calls = *result.tool_calls;
        message->mutable_tool_calls()->Swap(&source_tool_calls);
      }

      if (!result.finish_reason.empty()) {
        choice->mutable_finish_reason()->swap(result.finish_reason);
      }
    } else {
      set_output_and_finish_reason();
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  return call->write_and_finish(response);
}

}  // namespace

ChatServiceImpl::ChatServiceImpl(LLMMaster* master,
                                 const std::vector<std::string>& models)
    : APIServiceImpl(models),
      master_(master),
      parser_format_(master_->options().tool_call_parser().value_or("")) {
  CHECK(master_ != nullptr);
}

// chat_async for brpc
void ChatServiceImpl::process_async_impl(std::shared_ptr<ChatCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited.
  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  std::vector<Message> messages;
  messages.reserve(rpc_request.messages_size());
  for (const auto& message : rpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }
  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  const bool has_tool_support =
      !request_params.tools.empty() && !parser_format_.empty();

  std::shared_ptr<StreamingState> streaming_state;
  if (request_params.streaming && has_tool_support) {
    streaming_state =
        std::make_shared<StreamingState>(request_params.tools, parser_format_);
  }

  master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master_,
       stream = request_params.streaming,
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = request_params.request_id,
       created_time = absl::ToUnixSeconds(absl::Now()),
       json_tools = request_params.tools,
       parser_format = parser_format_,
       streaming_state = streaming_state,
       has_tool_support =
           has_tool_support](const RequestOutput& req_output) mutable -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a
            // request is finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request
        // is finished or canceled.
        if (req_output.finished || req_output.cancelled) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output,
                                           streaming_state);
        } else {
          return send_result_to_client_brpc(call,
                                            request_id,
                                            created_time,
                                            model,
                                            req_output,
                                            parser_format,
                                            json_tools);
        }
      });
}

MMChatServiceImpl::MMChatServiceImpl(VLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master != nullptr);
}

void MMChatServiceImpl::process_async_impl(std::shared_ptr<MMChatCall> call) {
  const auto& rpc_request = call->request();
  const auto& req_messages = rpc_request.messages();
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // check if the request image number exceeds the allowed image limit.
  int image_count = 0;
  const int image_limit = master_->get_image_limit();
  for (const auto& message : req_messages) {
    for (const auto& content_item : message.content()) {
      if (content_item.type() == "image_url") {
        ++image_count;
        if (image_count > image_limit) {
          call->finish_with_error(
              StatusCode::INVALID_ARGUMENT,
              "Number of images exceeds the allowed image limit.");
          return;
        }
      }
    }
  }
  // Check if the request is being rate-limited.
  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  std::vector<Message> messages;
  MMInput mm_inputs;

  static MMInputHelper helper;
  if (!helper.trans(req_messages, messages, mm_inputs.items_)) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "inputs argument is invalid.");
    return;
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  // schedule the request
  master_->handle_request(
      std::move(messages),
      std::move(mm_inputs),
      std::move(request_params),
      [call,
       model,
       master = master_,
       stream = request_params.streaming,
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = request_params.request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) mutable -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a request is
            // finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request is finished
        // or canceled.
        if (req_output.finished || req_output.cancelled) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          // send delta to client
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output);
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
