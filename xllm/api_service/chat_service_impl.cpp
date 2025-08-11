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
#include "core/framework/request/mm_input_helper.h"
#include "core/framework/request/request_params.h"
#include "core/runtime/llm_master.h"
#include "core/runtime/vlm_master.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "chat_template/chat_template.h"
#include "common/instance_name.h"
#include "common/uuid.h"
#include "function_call/function_call.h"
#include "function_call/core_types.h"
#include "request/request_params.h"
#include "util/utils.h"

namespace xllm {
namespace {


std::string generate_tool_call_id() { return "call_" + llm::generate_uuid(); }

struct ToolCallResult {
  std::optional<std::vector<proto::ToolCall>> tool_calls;
  std::string text;
  std::string finish_reason;
};

ToolCallResult process_tool_calls(
    const std::string& text,
    const std::vector<proto::Tool>& tools,
    const std::string& parser_format,
    const std::string& finish_reason) {

  ToolCallResult result;
  result.text = text;
  result.finish_reason = finish_reason;

  function_call::FunctionCallParser parser(tools, parser_format);

  if (!parser.has_tool_call(text)) {
    return result;
  }

  if (result.finish_reason == "stop") {
    result.finish_reason = "tool_calls";
  }

  try {
    auto [parsed_text, call_info_list] = parser.parse_non_stream(text);
    result.text = parsed_text;

    std::vector<proto::ToolCall> tool_calls;
    tool_calls.reserve(call_info_list.size());

    for (const auto& call_info : call_info_list) {
      proto::ToolCall tool_call;
      tool_call.set_id("call_" + generate_uuid());
      tool_call.set_type("function");

      auto* function = tool_call.mutable_function();
      if (call_info.name) {
        function->set_name(*call_info.name);
      }
      function->set_arguments(call_info.parameters);

      tool_calls.push_back(tool_call);
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

template <typename ChatCall>
bool send_delta_to_client_brpc(std::shared_ptr<ChatCall> call,
                               bool include_usage,
                               std::unordered_set<size_t>* first_message_sent,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model,
                               const RequestOutput& output) {
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

    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();
      choice->set_finish_reason(seq_output.finish_reason.value());
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
                                bool has_tools = false,
                                const std::string& parser_format = "",
                                const std::vector<proto::Tool>& tools = {}) {
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

    auto setOutputAndFinishReason = [&]() {
      message->set_content(output.text);
      if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
    };

    if (has_tools && !parser_format.empty()) {
      std::string finish_reason;
      if (output.finish_reason.has_value()) {
        finish_reason = output.finish_reason.value();
      }

      auto result = process_tool_calls(output.text, tools, parser_format, finish_reason);
      
      message->set_content(result.text);
      
      if (result.tool_calls) {
        for (const auto& tool_call : *result.tool_calls) {
          *message->add_tool_calls() = tool_call;
        }
      }
      
      if (!result.finish_reason.empty()) {
        choice->set_finish_reason(result.finish_reason);
      }
    } else {
      setOutputAndFinishReason();
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
    : APIServiceImpl(master, models) {}

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
    prompt_tokens->reserve(rpc_request.routing().token_ids_size());
    for (int i = 0; i < rpc_request.routing().token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.routing().token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
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


}  // namespace

ChatServiceImpl::ChatServiceImpl(LLMMaster* master,
                                 const std::vector<std::string>& models)
    : APIServiceImpl(master, models) {}

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
    prompt_tokens->reserve(rpc_request.routing().token_ids_size());
    for (int i = 0; i < rpc_request.routing().token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.routing().token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
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
                                           req_output);
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

MMChatServiceImpl::MMChatServiceImpl(VLMMaster* master,
                                     const std::vector<std::string>& models)
    : master_(master), models_(models.begin(), models.end()) {
  CHECK(master != nullptr);
  CHECK(!models_.empty());
}

void MMChatServiceImpl::process_async(std::shared_ptr<MMChatCall> call) {
  const auto& rpc_request = call->request();
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
  MMInput mm_inputs;

  MMInputHelper helper;
  if (!helper.trans(rpc_request.messages(), messages, mm_inputs.items_)) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "inputs argument is invalid.");
    return;
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  if (rpc_request.tools_size() > 0) {
    request_params.proto_tools.assign(rpc_request.tools().begin(),
                                      rpc_request.tools().end());

    // TODO: Implement support for 'required' option in tool_choice.
    if (rpc_request.has_tool_choice()) {
      request_params.tool_choice = rpc_request.tool_choice();
    } else {
      request_params.tool_choice = "auto";
    }
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
       created_time = absl::ToUnixSeconds(absl::Now()),
       has_tools = request_params.has_tools(),
       proto_tools = request_params.proto_tools](
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

        std::string parser_format =
            master->options().tool_call_parser().value_or("");
        if (stream) {
          if (has_tools && !parser_format.empty()) {
            LOG(ERROR) << "Tool call does not support streaming output";
          }
          // send delta to client
          return send_delta_to_client_brpc(call_data,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output);
        }

        if (has_tools && !parser_format.empty()) {
          return send_result_to_client_brpc(call_data,
                                            request_id,
                                            created_time,
                                            model,
                                            req_output,
                                            has_tools,
                                            parser_format,
                                            proto_tools);
        }

        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
