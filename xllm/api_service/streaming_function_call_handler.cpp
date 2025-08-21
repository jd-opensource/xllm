#include "streaming_function_call_handler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include "core/util/uuid.h"
#include "function_call/function_call.h"

namespace xllm {

StreamingFunctionCallHandler::StreamingFunctionCallHandler(
    const std::vector<function_call::JsonTool>& tools,
    const std::string& parser_format)
    : tools_(tools), parser_format_(parser_format) {
  if (!tools_.empty() && !parser_format_.empty()) {
    parser_ = std::make_unique<function_call::FunctionCallParser>(
        tools_, parser_format_);
  }
}

template <typename ChatCall>
bool StreamingFunctionCallHandler::process_streaming_output(
    std::shared_ptr<ChatCall> call,
    bool include_usage,
    std::unordered_set<size_t>* first_message_sent,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output) {
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;

    if (first_message_sent->find(index) == first_message_sent->end()) {
      auto& response = call->response();
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
      if (parser_) {
        if (!process_tool_call_stream(call,
                                      index,
                                      seq_output.text,
                                      request_id,
                                      created_time,
                                      model)) {
          return false;
        }
      } else {
        if (!send_normal_text_chunk(call,
                                    index,
                                    seq_output.text,
                                    request_id,
                                    created_time,
                                    model)) {
          return false;
        }
      }
    }

    if (seq_output.finish_reason.has_value()) {
      if (parser_ && has_tool_calls_[index]) {
        if (!check_for_unstreamed_tool_args(
                call, index, request_id, created_time, model)) {
          return false;
        }
      }

      auto& response = call->response();
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();

      std::string final_finish_reason = seq_output.finish_reason.value();
      if (has_tool_calls_[index] && final_finish_reason == "stop") {
        final_finish_reason = "tool_calls";
      }
      choice->set_finish_reason(final_finish_reason);

      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value() &&
      (output.finished || output.cancelled)) {
    auto& response = call->response();
    response.Clear();
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);

    const auto& usage = output.usage.value();
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

  return true;
}

template <typename ChatCall>
bool StreamingFunctionCallHandler::process_tool_call_stream(
    std::shared_ptr<ChatCall> call,
    size_t index,
    const std::string& delta,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model) {
  auto parse_result = parser_->parse_streaming_increment(delta);

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
    has_tool_calls_[index] = true;

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
bool StreamingFunctionCallHandler::check_for_unstreamed_tool_args(
    std::shared_ptr<ChatCall> call,
    size_t index,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model) {
  auto* detector = parser_->get_detector();
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
bool StreamingFunctionCallHandler::send_tool_call_chunk(
    std::shared_ptr<ChatCall> call,
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
bool StreamingFunctionCallHandler::send_normal_text_chunk(
    std::shared_ptr<ChatCall> call,
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

template bool StreamingFunctionCallHandler::process_streaming_output<ChatCall>(
    std::shared_ptr<ChatCall> call,
    bool include_usage,
    std::unordered_set<size_t>* first_message_sent,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output);

template bool StreamingFunctionCallHandler::process_streaming_output<
    MMChatCall>(std::shared_ptr<MMChatCall> call,
                bool include_usage,
                std::unordered_set<size_t>* first_message_sent,
                const std::string& request_id,
                int64_t created_time,
                const std::string& model,
                const RequestOutput& output);

}  // namespace xllm