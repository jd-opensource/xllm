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
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <filesystem>
#include <nlohmann/json.hpp>

#include "call.h"
#include "chat.pb.h"
#include "chat_json_utils.h"
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
  if (FLAGS_node_rank != 0) {
    masters_[model_names[0]] = master;
    return;
  }
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
  masters_[model_names[0]] = master;
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

  auto arena = GetArenaWithCheck<CompletionCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
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

size_t get_json_content_length(const brpc::Controller* ctrl) {
  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  if (infer_content_len != nullptr) {
    return std::stoul(*infer_content_len);
  }

  const auto content_len = ctrl->http_request().GetHeader(kContentLength);
  if (content_len != nullptr) {
    return std::stoul(*content_len);
  }

  LOG(ERROR) << "Content-Length header is missing.";
  return (size_t)-1L;
}

}  // namespace

// Preprocess chat JSON to normalize array content to string.
// For text-only backends, combines text array items into a single string.
// For multimodal backends, passes through unchanged without parsing.
// Returns Status with processed JSON on success, or error status on failure.
std::pair<Status, std::string> preprocess_chat_json(std::string json_str,
                                                    bool is_multimodal) {
  // Multimodal backends handle array content natively, skip parsing
  if (is_multimodal) {
    return {Status(), std::move(json_str)};
  }

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
        // Validate all items are text-only with proper text field
        for (const auto& item : msg["content"]) {
          if (!item.is_object()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Content array item must be an object."),
                    ""};
          }
          if (!item.contains("type") || item["type"] != "text") {
            // Non-text content on text-only backend is an error
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Non-text content (e.g., image_url) requires "
                           "multimodal backend (-backend vlm)"),
                    ""};
          }
          // Validate text items have proper text field
          if (!item.contains("text") || !item["text"].is_string()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Missing or invalid 'text' field in content item."),
                    ""};
          }
        }

        // All items are text-only; combine into single string.
        // Pre-calculate total size to avoid reallocations.
        size_t total_size = 0;
        size_t num_items = msg["content"].size();
        for (const auto& item : msg["content"]) {
          // Already validated above
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

namespace {

template <typename ChatCall, typename Service>
void chat_completions_impl(std::unique_ptr<Service>& service,
                           xllm::ClosureGuard& guard,
                           brpc::Controller* ctrl,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           bool is_multimodal) {
  auto arena = GetArenaWithCheck<ChatCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ReqType>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ResType>(arena);

  auto content_len = get_json_content_length(ctrl);
  if (content_len == (size_t)-1L) {
    ctrl->SetFailed("Content-Length header is missing.");
    return;
  }

  std::string attachment;
  ctrl->request_attachment().copy_to(&attachment, content_len, 0);

  auto [preprocess_status, processed_json] =
      preprocess_chat_json(std::move(attachment), is_multimodal);
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
    chat_completions_impl<ChatCall, ChatServiceImpl>(chat_service_impl_,
                                                     done_guard,
                                                     ctrl,
                                                     request,
                                                     response,
                                                     /*is_multimodal=*/false);
  } else if (FLAGS_backend == "vlm") {
    CHECK(mm_chat_service_impl_) << " mm chat service is invalid.";
    chat_completions_impl<MMChatCall, MMChatServiceImpl>(
        mm_chat_service_impl_,
        done_guard,
        ctrl,
        request,
        response,
        /*is_multimodal=*/true);
  } else if (FLAGS_backend == "rec") {
    CHECK(chat_service_impl_) << " chat service is invalid.";
    chat_completions_impl<ChatCall, ChatServiceImpl>(chat_service_impl_,
                                                     done_guard,
                                                     ctrl,
                                                     request,
                                                     response,
                                                     /*is_multimodal=*/false);
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
  auto arena = GetArenaWithCheck<EmbeddingCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ReqType>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename EmbeddingCall::ResType>(
          arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
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

  auto arena = GetArenaWithCheck<ImageGenerationCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::ImageGenerationResponse>(
          arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
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

  auto arena = GetArenaWithCheck<RerankCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RerankResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
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

bool APIService::ParseForkMasterRequest(const proto::MasterInfos* request,
                                        Options& options) {
  if (!std::filesystem::exists(request->model_path())) {
    LOG(ERROR) << "Model path " << request->model_path() << " does not exist.";
    return false;
  }

  std::filesystem::path model_path =
      std::filesystem::path(request->model_path()).lexically_normal();
  std::string model_id;
  if (model_path.has_filename()) {
    model_id = std::filesystem::path(request->model_path()).filename();
  } else {
    model_id =
        std::filesystem::path(request->model_path()).parent_path().filename();
  }
  options.model_id() = model_id;
  options.master_node_addr() = request->master_node_addr();
  options.model_path() = request->model_path();
  options.master_status() = request->master_status();

  // Parse nnodes and dp_size (tp_size = nnodes / dp_size, computed by engine)
  if (request->nnodes() > 0) {
    options.nnodes() = request->nnodes();
  }
  if (request->dp_size() > 0) {
    options.dp_size() = request->dp_size();
  }

  return true;
}

void APIService::ForkMaster(::google::protobuf::RpcController* controller,
                            const proto::MasterInfos* request,
                            proto::RpcStatus* response,
                            ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::ForkMasterHttp(::google::protobuf::RpcController* controller,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (FLAGS_backend != "llm") {
    LOG(ERROR) << "fork master only supports llm backend";
    return;
  }

  Options master_options;
  if (!ParseForkMasterRequest(req_pb, master_options)) {
    LOG(ERROR) << "Failed to parse fork master request";
    return;
  }

  if (masters_.find(master_options.model_id()) != masters_.end()) {
    LOG(INFO) << "Master for model " << master_options.model_id()
              << " already exists";
    return;
  }

  auto master = fork_master(master_, master_options);
  if (!master) {
    LOG(ERROR) << "Failed to fork master: " << master_options.model_id();
    return;
  }

  // CAS: only succeed if num_concurrent_requests == 0.
  if (master->get_master_status() != WAKEUP &&
      !master->get_rate_limiter()->try_set_sleeping()) {
    int32_t num_requests =
        master->get_rate_limiter()->get_num_concurrent_requests();
    LOG(ERROR) << "Cannot sleep model " << req_pb->model_id() << " with "
               << num_requests << " in-flight requests";
    ctrl->SetFailed("Cannot sleep model with in-flight requests");
    return;
  }

  masters_[master_options.model_id()] = master.get();
  if (FLAGS_node_rank == 0) {
    auto llm_master = dynamic_cast<LLMMaster*>(master.get());
    completion_service_impl_->add_model_master(master_options.model_id(),
                                               llm_master);
    chat_service_impl_->add_model_master(master_options.model_id(), llm_master);
  }
  master.release();
}

void APIService::Sleep(::google::protobuf::RpcController* controller,
                       const proto::MasterInfos* request,
                       proto::RpcStatus* response,
                       ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::SleepHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (req_pb->master_status() != LIGHT_SLEEP &&
      req_pb->master_status() != DEEP_SLEEP) {
    LOG(ERROR) << "Invalid sleep status: " << req_pb->master_status();
    ctrl->SetFailed("Invalid sleep status");
    return;
  }

  if (masters_.find(req_pb->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  auto master = masters_[req_pb->model_id()];
  if (master->get_master_status()) {
    LOG(INFO) << "Master for model " << req_pb->model_id()
              << " is already sleeping";
    ctrl->SetFailed("Master for model is already sleeping");
    return;
  }

  // CAS: only succeed if num_concurrent_requests == 0.
  if (!master->get_rate_limiter()->try_set_sleeping()) {
    int32_t num_requests =
        master->get_rate_limiter()->get_num_concurrent_requests();
    LOG(ERROR) << "Cannot sleep model " << req_pb->model_id() << " with "
               << num_requests << " in-flight requests";
    ctrl->SetFailed("Cannot sleep model with in-flight requests");
    return;
  }

  auto master_status = master->get_master_status();
  master->set_master_status(req_pb->master_status());
  if (!master->sleep()) {
    master->set_master_status(master_status);
    LOG(ERROR) << "Failed to sleep model " << req_pb->model_id();
    ctrl->SetFailed("Failed to sleep model");
    return;
  }
  // Success: return HTTP 200 with empty body
}

void APIService::Wakeup(::google::protobuf::RpcController* controller,
                        const proto::MasterInfos* request,
                        proto::RpcStatus* response,
                        ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::WakeupHttp(::google::protobuf::RpcController* controller,
                            const proto::HttpRequest* request,
                            proto::HttpResponse* response,
                            ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }
  if (masters_.find(req_pb->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  auto master = masters_[req_pb->model_id()];
  if (!master->get_master_status()) {
    LOG(INFO) << "Master for model " << req_pb->model_id()
              << " is already awake";
    ctrl->SetFailed("Master for model is already awake");
    return;
  }

  // Check if remote weight transfer is requested
  if (req_pb->remote_addrs_size() > 0) {
    WakeupOptions wakeup_options;
    wakeup_options.remote_addrs.assign(req_pb->remote_addrs().begin(),
                                       req_pb->remote_addrs().end());
    if (req_pb->src_weight_segments_size() > 0) {
      for (const auto& seg_list : req_pb->src_weight_segments()) {
        std::vector<WeightSegment> segments;
        segments.reserve(seg_list.segments_size());
        for (const auto& proto_seg : seg_list.segments()) {
          segments.push_back({proto_seg.offset(), proto_seg.size()});
        }
        wakeup_options.src_weight_segments.push_back(std::move(segments));
      }
    }
    if (!master->wakeup(wakeup_options)) {
      LOG(ERROR) << "Failed to wakeup model " << req_pb->model_id()
                 << " with remote weight transfer";
      ctrl->SetFailed("Failed to wakeup model with remote weight transfer");
      return;
    }
  } else {
    if (!master->wakeup()) {
      LOG(ERROR) << "Failed to wakeup model " << req_pb->model_id();
      ctrl->SetFailed("Failed to wakeup model");
      return;
    }
  }

  // Restore rate limiter from sleeping state
  if (!master->get_rate_limiter()->try_wakeup()) {
    LOG(ERROR) << "Failed to restore rate limiter for model "
               << req_pb->model_id();
    ctrl->SetFailed("Failed to restore rate limiter");
    return;
  }

  master->set_master_status(WAKEUP);
  // Success: return HTTP 200 with empty body
}

void APIService::LinkD2D(::google::protobuf::RpcController* controller,
                         const proto::D2DLinkRequest* request,
                         proto::RpcStatus* response,
                         ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (masters_.find(request->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_status(false);
    return;
  }

  auto master = masters_[request->model_id()];
  bool status = master->link_d2d(
      {request->device_ips().begin(), request->device_ips().end()});
  response->set_status(status);
}

void APIService::LinkD2DHttp(::google::protobuf::RpcController* controller,
                             const proto::HttpRequest* request,
                             proto::HttpResponse* response,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::D2DLinkRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (masters_.find(req_pb->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  auto master = masters_[req_pb->model_id()];
  bool status = master->link_d2d(
      {req_pb->device_ips().begin(), req_pb->device_ips().end()});
  resp_pb->set_status(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

void APIService::UnlinkD2D(::google::protobuf::RpcController* controller,
                           const proto::D2DLinkRequest* request,
                           proto::RpcStatus* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (masters_.find(request->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_status(false);
    return;
  }

  auto master = masters_[request->model_id()];
  bool status = master->unlink_d2d(
      {request->device_ips().begin(), request->device_ips().end()});
  response->set_status(status);
}

void APIService::UnlinkD2DHttp(::google::protobuf::RpcController* controller,
                               const proto::HttpRequest* request,
                               proto::HttpResponse* response,
                               ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::D2DLinkRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  auto st = json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (masters_.find(req_pb->model_id()) == masters_.end()) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  auto master = masters_[req_pb->model_id()];
  bool status = master->unlink_d2d(
      {req_pb->device_ips().begin(), req_pb->device_ips().end()});
  resp_pb->set_status(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

}  // namespace xllm
