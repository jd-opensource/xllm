#include "request_params.h"

#include "core/common/instance_name.h"
#include "core/util/uuid.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_completion_request_id() {
  return "cmpl-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::string generate_embedding_request_id() {
  return "embeddingcmpl-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::string generate_chat_request_id() {
  return "chatcmpl-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

}  // namespace

nlohmann::json RequestParams::proto_value_to_json(
    const google::protobuf::Value& pb_value) {
  switch (pb_value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nlohmann::json(nullptr);

    case google::protobuf::Value::kNumberValue:
      return nlohmann::json(pb_value.number_value());

    case google::protobuf::Value::kStringValue:
      return nlohmann::json(pb_value.string_value());

    case google::protobuf::Value::kBoolValue:
      return nlohmann::json(pb_value.bool_value());

    case google::protobuf::Value::kStructValue:
      return proto_struct_to_json(pb_value.struct_value());

    case google::protobuf::Value::kListValue: {
      nlohmann::json array = nlohmann::json::array();
      const auto& list = pb_value.list_value();
      for (const auto& item : list.values()) {
        array.push_back(proto_value_to_json(item));
      }
      return array;
    }

    case google::protobuf::Value::KIND_NOT_SET:
    default:
      return nlohmann::json(nullptr);
  }
}

nlohmann::json RequestParams::proto_struct_to_json(
    const google::protobuf::Struct& pb_struct) {
  nlohmann::json result = nlohmann::json::object();

  for (const auto& field : pb_struct.fields()) {
    result[field.first] = proto_value_to_json(field.second);
  }

  return result;
}

void RequestParams::parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<proto::Tool>& proto_tools) {
  tools.clear();
  tools.reserve(proto_tools.size());

  for (const auto& proto_tool : proto_tools) {
    function_call::JsonTool json_tool;
    json_tool.type = proto_tool.type();

    const auto& proto_function = proto_tool.function();
    json_tool.function.name = proto_function.name();
    json_tool.function.description = proto_function.description();

    if (proto_function.has_parameters()) {
      json_tool.function.parameters =
          proto_struct_to_json(proto_function.parameters());
    } else {
      json_tool.function.parameters = nlohmann::json::object();
    }

    tools.emplace_back(std::move(json_tool));
  }
}

RequestParams::RequestParams(const proto::CompletionRequest& request,
                             const std::string& x_rid,
                             const std::string& x_rtime) {
  request_id = generate_completion_request_id();
  x_request_id = x_rid;
  x_request_time = x_rtime;

  if (request.has_service_request_id()) {
    service_request_id = request.service_request_id();
  }
  if (request.has_max_tokens()) {
    max_tokens = request.max_tokens();
  }
  if (request.has_n()) {
    n = request.n();
  }
  if (request.has_best_of()) {
    best_of = request.best_of();
  }
  if (request.has_echo()) {
    echo = request.echo();
  }
  if (request.has_frequency_penalty()) {
    frequency_penalty = request.frequency_penalty();
  }
  if (request.has_presence_penalty()) {
    presence_penalty = request.presence_penalty();
  }
  if (request.has_repetition_penalty()) {
    repetition_penalty = request.repetition_penalty();
  }
  if (request.has_temperature()) {
    temperature = request.temperature();
  }
  if (request.has_top_p()) {
    top_p = request.top_p();
  }
  if (request.has_top_k()) {
    top_k = request.top_k();
  }
  if (request.has_logprobs()) {
    logprobs = true;
    top_logprobs = request.logprobs();
  }
  if (request.has_skip_special_tokens()) {
    skip_special_tokens = request.skip_special_tokens();
  }
  if (request.has_ignore_eos()) {
    ignore_eos = request.ignore_eos();
  }
  if (request.stop_size() > 0) {
    stop =
        std::vector<std::string>(request.stop().begin(), request.stop().end());
  }
  if (request.stop_token_ids_size() > 0) {
    stop_token_ids = std::vector<int32_t>(request.stop_token_ids().begin(),
                                          request.stop_token_ids().end());
  }
  if (request.has_stream()) {
    const size_t best_of_value = best_of.value_or(n);
    if (request.stream() && best_of_value == n) {
      streaming = true;
    } else {
      streaming = false;
    }
  }
}

namespace {
template <typename ChatRequest>
void InitFromChatRequest(RequestParams& params, const ChatRequest& request) {
  if (request.has_request_id()) {
    params.request_id = request.request_id();
  }
  if (request.has_service_request_id()) {
    params.service_request_id = request.service_request_id();
  }
  if (request.has_max_tokens()) {
    params.max_tokens = request.max_tokens();
  }
  if (request.has_n()) {
    params.n = request.n();
  }
  if (request.has_best_of()) {
    params.best_of = request.best_of();
  }
  if (request.has_frequency_penalty()) {
    params.frequency_penalty = request.frequency_penalty();
  }
  if (request.has_presence_penalty()) {
    params.presence_penalty = request.presence_penalty();
  }
  if (request.has_repetition_penalty()) {
    params.repetition_penalty = request.repetition_penalty();
  }
  if (request.has_temperature()) {
    params.temperature = request.temperature();
  }
  if (request.has_top_p()) {
    params.top_p = request.top_p();
  }
  if (request.has_top_k()) {
    params.top_k = request.top_k();
  }
  if (request.has_logprobs()) {
    params.logprobs = request.logprobs();
  }
  if (request.has_top_logprobs()) {
    params.top_logprobs = request.top_logprobs();
  }
  if (request.has_skip_special_tokens()) {
    params.skip_special_tokens = request.skip_special_tokens();
  }
  if (request.has_ignore_eos()) {
    params.ignore_eos = request.ignore_eos();
  }
  if (request.stop_size() > 0) {
    params.stop =
        std::vector<std::string>(request.stop().begin(), request.stop().end());
  }
  if (request.stop_token_ids_size() > 0) {
    params.stop_token_ids = std::vector<int32_t>(
        request.stop_token_ids().begin(), request.stop_token_ids().end());
  }
  if (request.has_stream()) {
    const size_t best_of_value = params.best_of.value_or(params.n);
    if (request.stream() && best_of_value == params.n) {
      params.streaming = true;
    } else {
      params.streaming = false;
    }
  }

  // Parse tools from proto request
  if (request.tools_size() > 0) {
    parse_tools_from_proto(request.tools());

    if (request.has_tool_choice()) {
      tool_choice = request.tool_choice();
    } else {
      tool_choice = "auto";
    }
  }
}
}  // namespace

RequestParams::RequestParams(const proto::ChatRequest& request,
                             const std::string& x_rid,
                             const std::string& x_rtime) {
  request_id = generate_chat_request_id();
  x_request_id = x_rid;
  x_request_time = x_rtime;

  InitFromChatRequest(*this, request);
}

RequestParams::RequestParams(const proto::MMChatRequest& request,
                             const std::string& x_rid,
                             const std::string& x_rtime) {
  request_id = generate_chat_request_id();
  x_request_id = x_rid;
  x_request_time = x_rtime;

  InitFromChatRequest(*this, request);
}

RequestParams::RequestParams(const proto::EmbeddingRequest& request,
                             const std::string& x_rid,
                             const std::string& x_rtime) {
  request_id = generate_embedding_request_id();
  if (request.has_service_request_id()) {
    service_request_id = request.service_request_id();
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;
  is_embeddings = true;
  max_tokens = 1;
  streaming = false;
}

bool RequestParams::verify_params(OutputCallback callback) const {
  if (n == 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "n should be greater than 0");
    return false;
  }
  if (best_of.has_value()) {
    if (n > best_of.value()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "n should be less than or equal to best_of");
      return false;
    }
  }

  // up to 4 stop sequences
  if (stop.has_value() && stop.value().size() > 4) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "stop size is too large");
    return false;
  }

  // temperature between [0.0, 2.0]
  if (temperature < 0.0 || temperature > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "temperature must be between 0.0 and 2.0");
    return false;
  }

  // top_p between [0.0, 1.0]
  if (top_p < 0.0 || top_p > 1.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "top_p must be between 0.0 and 1.0");
    return false;
  }

  if (logprobs) {
    if (echo) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "logprobs is not supported with echo");
      return false;
    }
    if (top_logprobs < 0 || top_logprobs > 20) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "logprobs must be between 0 and 20");
      return false;
    }
  }

  // presence_penalty between [-2.0, 2.0]
  if (presence_penalty < -2.0 || presence_penalty > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "presence_penalty must be between -2.0 and 2.0");
    return false;
  }

  // frequency_penalty between [0.0, 2.0]
  if (frequency_penalty < 0.0 || frequency_penalty > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "frequency_penalty must be between 0.0 and 2.0");
    return false;
  }
  return true;
}

}  // namespace xllm
