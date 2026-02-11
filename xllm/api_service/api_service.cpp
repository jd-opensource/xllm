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

#include "api_service.h"

#include <glog/logging.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <charconv>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>

#include "call.h"
#include "chat.pb.h"
#include "common.pb.h"
#include "completion.pb.h"
#include "core/common/constants.h"
#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/util/closure_guard.h"
#include "embedding.pb.h"
#include "image_generation.pb.h"
#include "models.pb.h"
#include "service_impl_factory.h"
#include "xllm_metrics.h"
namespace xllm {

namespace {

std::optional<size_t> GetJsonContentLength(const brpc::Controller* ctrl) {
  const auto* header_val = ctrl->http_request().GetHeader(kInferContentLength);
  if (header_val == nullptr) {
    header_val = ctrl->http_request().GetHeader(kContentLength);
  }

  if (header_val == nullptr) {
    LOG(WARNING) << "Content-Length header is missing";
    return std::nullopt;
  }

  if (header_val->empty()) {
    LOG(WARNING) << "Content-Length header is empty";
    return std::nullopt;
  }

  size_t length = 0;
  const auto* begin = header_val->data();
  const auto* end = begin + header_val->size();
  auto [ptr, ec] = std::from_chars(begin, end, length);
  if (ec != std::errc{} || ptr != end) {
    LOG(WARNING) << "Invalid Content-Length header: " << *header_val;
    return std::nullopt;
  }

  return length;
}

// Validates Content-Length header. Returns validated content length, or
// nullopt if validation fails (also sets appropriate HTTP error response).
// Note: brpc's -max_body_size flag provides payload size protection.
std::optional<size_t> ValidateContentLength(brpc::Controller* ctrl) {
  auto content_len_opt = GetJsonContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    ctrl->SetFailed("Missing or invalid Content-Length header");
    ctrl->http_response().set_status_code(400);
    return std::nullopt;
  }
  return content_len_opt.value();
}

template <typename Call>
google::protobuf::Arena* GetArenaWithCheck(
    const google::protobuf::Message* message) {
  if (xllm::is_stream_call_v<Call>) {
    return nullptr;
  } else {
    return message->GetArena();
  }
}
}  // namespace

APIService::APIService(Master* master,
                       const std::vector<std::string>& model_names,
                       const std::vector<std::string>& model_versions)
    : master_(master) {
  if (FLAGS_backend == "llm") {
    auto llm_master = dynamic_cast<LLMMaster*>(master);
    anthropic_service_impl_ =
        std::make_unique<AnthropicServiceImpl>(llm_master, model_names);
    completion_service_impl_ =
        ServiceImplFactory<CompletionServiceImpl>::create_service_impl(
            llm_master, model_names);
    chat_service_impl_ =
        ServiceImplFactory<ChatServiceImpl>::create_service_impl(llm_master,
                                                                 model_names);
    embedding_service_impl_ =
        ServiceImplFactory<EmbeddingServiceImpl>::create_service_impl(
            llm_master, model_names);
    if (FLAGS_enable_qwen3_reranker) {
      rerank_service_impl_ =
          ServiceImplFactory<Qwen3RerankServiceImpl>::create_service_impl(
              llm_master, model_names);
    } else {
      rerank_service_impl_ =
          ServiceImplFactory<RerankServiceImpl>::create_service_impl(
              llm_master, model_names);
    }
  } else if (FLAGS_backend == "vlm") {
    auto vlm_master = dynamic_cast<VLMMaster*>(master);
    mm_chat_service_impl_ =
        std::make_unique<MMChatServiceImpl>(vlm_master, model_names);
    mm_embedding_service_impl_ =
        std::make_unique<MMEmbeddingServiceImpl>(vlm_master, model_names);
  } else if (FLAGS_backend == "dit") {
    image_generation_service_impl_ =
        std::make_unique<ImageGenerationServiceImpl>(
            dynamic_cast<DiTMaster*>(master), model_names);
  } else if (FLAGS_backend == "rec") {
    auto rec_master = dynamic_cast<RecMaster*>(master);
    rec_completion_service_impl_ =
        std::make_unique<RecCompletionServiceImpl>(rec_master, model_names);
    chat_service_impl_ =
        std::make_unique<ChatServiceImpl>(rec_master, model_names);
  }
  models_service_impl_ =
      ServiceImplFactory<ModelsServiceImpl>::create_service_impl(
          model_names, model_versions);
}

void APIService::Completions(::google::protobuf::RpcController* controller,
                             const proto::CompletionRequest* request,
                             proto::CompletionResponse* response,
                             ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null.";
    return;
  }
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  auto arena = GetArenaWithCheck<CompletionCall>(response);
  std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
      ctrl,
      done_guard.release(),
      const_cast<proto::CompletionRequest*>(request),
      response,
      arena != nullptr);
  if (FLAGS_backend == "llm") {
    completion_service_impl_->process_async(call);
  } else if (FLAGS_backend == "rec") {
    rec_completion_service_impl_->process_async(call);
  }
}

void APIService::CompletionsHttp(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  auto content_len_opt = ValidateContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    return;
  }
  auto content_len = content_len_opt.value();

  auto arena = GetArenaWithCheck<CompletionCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionResponse>(arena);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf body_buf;
  ctrl->request_attachment().cutn(&body_buf, content_len);
  butil::IOBufAsZeroCopyInputStream iobuf_stream(body_buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  if (FLAGS_backend == "llm") {
    completion_service_impl_->process_async(call);
  } else if (FLAGS_backend == "rec") {
    rec_completion_service_impl_->process_async(call);
  }
}

void APIService::ChatCompletions(::google::protobuf::RpcController* controller,
                                 const proto::ChatRequest* request,
                                 proto::ChatResponse* response,
                                 ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

namespace {

// Preprocess chat JSON to normalize array content to string.
// Returns Status with processed JSON on success, or error status on failure.
std::pair<Status, std::string> pre_process_chat_json(std::string json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);
    if (!json.contains("messages") || !json["messages"].is_array()) {
      return {Status(), std::move(json_str)};
    }

    bool modified = false;
    for (auto& msg : json["messages"]) {
      if (!msg.is_object()) {
        return {Status(StatusCode::INVALID_ARGUMENT,
                       "Message in 'messages' array must be an object."),
                ""};
      }
      if (msg.contains("content") && msg["content"].is_array()) {
        // Pre-calculate total size to avoid multiple reallocations
        size_t total_size = 0;
        size_t num_items = msg["content"].size();
        for (const auto& item : msg["content"]) {
          if (!item.is_object()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Content array item must be an object."),
                    ""};
          }
          if (!item.contains("type") || item["type"] != "text") {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Non-text content requires multimodal endpoint"),
                    ""};
          }
          if (!item.contains("text")) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Missing 'text' field for content item with type "
                           "'text'."),
                    ""};
          }
          if (!item["text"].is_string()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Invalid 'text' field in content item: must be a "
                           "string."),
                    ""};
          }
          total_size += item["text"].get_ref<const std::string&>().size();
        }
        // Add space for newline separators
        if (num_items > 1) {
          total_size += num_items - 1;
        }

        // Reserve capacity once to avoid reallocations
        std::string combined_text;
        combined_text.reserve(total_size);
        bool first = true;
        for (const auto& item : msg["content"]) {
          if (!first) {
            combined_text += '\n';
          }
          combined_text += item["text"].get_ref<const std::string&>();
          first = false;
        }
        msg["content"] = combined_text;
        modified = true;
      }
    }
    return modified ? std::make_pair(Status(), json.dump())
                    : std::make_pair(Status(), std::move(json_str));
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during JSON preprocessing: " << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

template <typename ChatCall, typename Service>
void handle_chat_completions(std::unique_ptr<Service>& service,
                             xllm::ClosureGuard& guard,
                             brpc::Controller* ctrl,
                             const proto::HttpRequest* request,
                             proto::HttpResponse* response) {
  auto arena = GetArenaWithCheck<ChatCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ReqType>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ResType>(arena);

  auto content_len_opt = ValidateContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    return;
  }
  auto content_len = content_len_opt.value();

  std::string attachment;
  ctrl->request_attachment().copy_to(&attachment, content_len, 0);

  auto [preprocess_status, processed_json] =
      pre_process_chat_json(std::move(attachment));
  if (!preprocess_status.ok()) {
    ctrl->SetFailed(preprocess_status.message());
    LOG(ERROR) << "Complex message preprocessing failed: "
               << preprocess_status.message();
    return;
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      processed_json, req_pb, options);
  if (!status.ok()) {
    ctrl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto call = std::make_shared<ChatCall>(
      ctrl, guard.release(), req_pb, resp_pb, arena != nullptr /*use_arena*/);
  service->process_async(call);
}
}  // namespace

void APIService::ChatCompletionsHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (FLAGS_backend == "llm") {
    CHECK(chat_service_impl_) << " chat service is invalid.";
    handle_chat_completions<ChatCall, ChatServiceImpl>(
        chat_service_impl_, done_guard, ctrl, request, response);
  } else if (FLAGS_backend == "vlm") {
    CHECK(mm_chat_service_impl_) << " mm chat service is invalid.";
    handle_chat_completions<MMChatCall, MMChatServiceImpl>(
        mm_chat_service_impl_, done_guard, ctrl, request, response);
  } else if (FLAGS_backend == "rec") {
    CHECK(chat_service_impl_) << " chat service is invalid.";
    handle_chat_completions<ChatCall, ChatServiceImpl>(
        chat_service_impl_, done_guard, ctrl, request, response);
  }
}

void APIService::Embeddings(::google::protobuf::RpcController* controller,
                            const proto::EmbeddingRequest* request,
                            proto::EmbeddingResponse* response,
                            ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

namespace {
template <typename EmbeddingCall, typename Service>
void handle_embedding_request(std::unique_ptr<Service>& embedding_service_impl_,
                              ::google::protobuf::RpcController* controller,
                              const proto::HttpRequest* request,
                              proto::HttpResponse* response,
                              ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  auto content_len_opt = ValidateContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    return;
  }
  auto content_len = content_len_opt.value();

  auto arena = GetArenaWithCheck<EmbeddingCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ReqType>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ResType>(
          arena);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf body_buf;
  ctrl->request_attachment().cutn(&body_buf, content_len);
  butil::IOBufAsZeroCopyInputStream iobuf_stream(body_buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  // default set to "float"
  if (req_pb->encoding_format().empty()) {
    req_pb->set_encoding_format("float");
  }

  std::shared_ptr<Call> call = std::make_shared<EmbeddingCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  embedding_service_impl_->process_async(call);
}
}  // namespace

void APIService::EmbeddingsHttp(::google::protobuf::RpcController* controller,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                ::google::protobuf::Closure* done) {
  if (FLAGS_backend == "llm") {
    CHECK(embedding_service_impl_) << " embedding service is invalid.";
    handle_embedding_request<EmbeddingCall, EmbeddingServiceImpl>(
        embedding_service_impl_, controller, request, response, done);
  } else if (FLAGS_backend == "vlm") {
    CHECK(mm_embedding_service_impl_) << " mm embedding service is invalid.";
    handle_embedding_request<MMEmbeddingCall, MMEmbeddingServiceImpl>(
        mm_embedding_service_impl_, controller, request, response, done);
  }
}

void APIService::ImageGeneration(::google::protobuf::RpcController* controller,
                                 const proto::ImageGenerationRequest* request,
                                 proto::ImageGenerationResponse* response,
                                 ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::ImageGenerationHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  auto content_len_opt = ValidateContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    return;
  }
  auto content_len = content_len_opt.value();

  auto arena = GetArenaWithCheck<ImageGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationResponse>(
          arena);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf body_buf;
  ctrl->request_attachment().cutn(&body_buf, content_len);
  butil::IOBufAsZeroCopyInputStream iobuf_stream(body_buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  std::shared_ptr<ImageGenerationCall> call =
      std::make_shared<ImageGenerationCall>(
          ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  image_generation_service_impl_->process_async(call);
}

void APIService::Rerank(::google::protobuf::RpcController* controller,
                        const proto::RerankRequest* request,
                        proto::RerankResponse* response,
                        ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::RerankHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  auto content_len_opt = ValidateContentLength(ctrl);
  if (!content_len_opt.has_value()) {
    return;
  }
  auto content_len = content_len_opt.value();

  auto arena = GetArenaWithCheck<RerankCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankResponse>(arena);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf body_buf;
  ctrl->request_attachment().cutn(&body_buf, content_len);
  butil::IOBufAsZeroCopyInputStream iobuf_stream(body_buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<RerankCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  rerank_service_impl_->process_async(call);
}

void APIService::Models(::google::protobuf::RpcController* controller,
                        const proto::ModelListRequest* request,
                        proto::ModelListResponse* response,
                        ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::ModelsHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ModelList>(arena);

  proto::ModelListResponse model_list;
  bool st_models = models_service_impl_->list_models(nullptr, &model_list);
  if (!st_models) {
    LOG(ERROR) << "list models failed.";
    return;
  }
  resp_pb->mutable_data()->CopyFrom(model_list.data());
  resp_pb->set_object("list");

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed";
    return;
  }
}

void APIService::ModelVersionsHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  ctrl->response_attachment().append(
      models_service_impl_->list_model_versions());

  return;
}

namespace {

// Preprocess Anthropic API JSON to convert "content" field to
// protobuf-compatible format Anthropic API uses "content" field which can be
// string or array Our protobuf uses "content_string" for string and
// "content_blocks" for array
std::string preprocess_anthropic_json(const std::string& json_str) {
  try {
    nlohmann::json j = nlohmann::json::parse(json_str);

    if (j.contains("messages") && j["messages"].is_array()) {
      for (auto& msg : j["messages"]) {
        if (msg.contains("content")) {
          auto& content = msg["content"];
          if (content.is_string()) {
            // Convert "content": "string" to "content_string": "string"
            msg["content_string"] = content.get<std::string>();
            msg.erase("content");
          } else if (content.is_array()) {
            // Convert "content": [...] to "content_blocks": {"blocks": [...]}
            nlohmann::json content_blocks;
            content_blocks["blocks"] = content;
            msg["content_blocks"] = content_blocks;
            msg.erase("content");
          }
        }
      }
    }

    if (j.contains("system")) {
      auto& system = j["system"];
      if (system.is_string()) {
        j["system_string"] = system.get<std::string>();
        j.erase("system");
      } else if (system.is_array()) {
        nlohmann::json system_blocks;
        system_blocks["blocks"] = system;
        j["system_blocks"] = system_blocks;
        j.erase("system");
      }
    }

    return j.dump();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to preprocess Anthropic JSON: " << e.what();
    return json_str;  // Return original on error
  }
}

void handle_anthropic_messages(std::unique_ptr<AnthropicServiceImpl>& service,
                               xllm::ClosureGuard& guard,
                               brpc::Controller* ctrl,
                               const proto::HttpRequest* request,
                               proto::HttpResponse* response) {
  auto arena = GetArenaWithCheck<AnthropicCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename AnthropicCall::ReqType>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename AnthropicCall::ResType>(
          arena);

  auto content_len = get_json_content_length(ctrl);
  if (content_len == (size_t)-1L) {
    ctrl->SetFailed("Content-Length header is missing.");
    return;
  }
  std::string attachment;
  ctrl->request_attachment().copy_to(&attachment, content_len, 0);

  // Preprocess JSON to convert Anthropic API format to protobuf-compatible
  // format
  std::string processed_json = preprocess_anthropic_json(attachment);

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      processed_json, req_pb, options);
  if (!status.ok()) {
    ctrl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto call = std::make_shared<AnthropicCall>(
      ctrl, guard.release(), req_pb, resp_pb, arena != nullptr /*use_arena*/);

  service->process_async(call);
}

}  // namespace

void APIService::AnthropicMessagesHttp(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (FLAGS_backend == "llm") {
    CHECK(anthropic_service_impl_) << " anthropic service is invalid.";
    handle_anthropic_messages(
        anthropic_service_impl_, done_guard, ctrl, request, response);
  } else {
    ctrl->SetFailed("Anthropic messages API is only supported for LLM backend");
    LOG(ERROR) << "Anthropic messages API is only supported for LLM backend";
  }
}

}  // namespace xllm
