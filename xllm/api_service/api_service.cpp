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

#include "api_service/chat_json_parser.h"
#include "api_service/service_impl_factory.h"
#include "api_service/serving_mode.h"
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
template <typename Call>
google::protobuf::Arena* GetArenaWithCheck(
    const google::protobuf::Message* message) {
  if (xllm::is_stream_call_v<Call>) {
    return nullptr;
  } else {
    return message->GetArena();
  }
}

const char* kSampleNotSupportedError = "/v1/sample is only supported for LLM";

}  // namespace

APIService::APIService(Master* master,
                       const std::vector<std::string>& model_names,
                       const std::vector<std::string>& model_versions)
    : master_(master) {
  set_model_master(model_names[0], master);
  if (FLAGS_node_rank != 0) {
    return;
  }
  ServiceImplFactory::create(this, master, model_names, model_versions);
  register_chat_completions_handler();
}

void APIService::set_model_master(const std::string& model_id, Master* master) {
  std::unique_lock<std::shared_mutex> lock(masters_mutex_);
  masters_.insert_or_assign(model_id, master);
}

bool APIService::has_model_master(const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(masters_mutex_);
  return masters_.find(model_id) != masters_.end();
}

bool APIService::add_model_master_if_absent(const std::string& model_id,
                                            Master* master) {
  std::unique_lock<std::shared_mutex> lock(masters_mutex_);
  return masters_.emplace(model_id, master).second;
}

Master* APIService::get_model_master(const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(masters_mutex_);
  auto it = masters_.find(model_id);
  if (it == masters_.end()) {
    return nullptr;
  }
  return it->second;
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

  if (completion_service_impl_) {
    completion_service_impl_->process_async_rpc_impl(request);
  } else if (rec_completion_service_impl_) {
    auto arena = GetArenaWithCheck<CompletionCall>(response);
    std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
        ctrl,
        done_guard.release(),
        const_cast<proto::CompletionRequest*>(request),
        response,
        arena != nullptr);
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
  if (completion_service_impl_) {
    completion_service_impl_->process_async(call);
  } else if (rec_completion_service_impl_) {
    rec_completion_service_impl_->process_async(call);
  }
}

void APIService::Sample(::google::protobuf::RpcController* controller,
                        const proto::SampleRequest* request,
                        proto::SampleResponse* response,
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
  if (!sample_service_impl_) {
    ctrl->SetFailed(kSampleNotSupportedError);
    return;
  }

  Status status;
  if (!sample_service_impl_->process_request(*request, response, &status)) {
    ctrl->SetFailed(status.message());
    LOG(ERROR) << "sample request failed: " << status.message();
  }
}

void APIService::SampleHttp(::google::protobuf::RpcController* controller,
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
  if (!sample_service_impl_) {
    ctrl->SetFailed(kSampleNotSupportedError);
    return;
  }

  auto arena = GetArenaWithCheck<SampleCall>(response);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::SampleRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::SampleResponse>(arena);

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

  std::shared_ptr<Call> call = std::make_shared<SampleCall>(
      ctrl, done_guard.release(), req_pb, resp_pb, arena != nullptr);
  sample_service_impl_->process_async(call);
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

namespace {

template <typename ChatCall, typename Service>
void chat_completions_http_impl(std::unique_ptr<Service>& service,
                                xllm::ClosureGuard& guard,
                                brpc::Controller* ctrl,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                const ChatJsonParser& chat_json_parser) {
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
      chat_json_parser.preprocess(std::move(attachment));
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

void APIService::register_chat_completions_handler() {
  if (mm_chat_service_impl_) {
    chat_completions_handler_ = [this](ClosureGuard& guard,
                                       brpc::Controller* ctrl,
                                       const proto::HttpRequest* request,
                                       proto::HttpResponse* response) {
      chat_completions_http_impl<MMChatCall, MMChatServiceImpl>(
          mm_chat_service_impl_,
          guard,
          ctrl,
          request,
          response,
          ChatJsonParser::get(ServingMode::VLM));
    };
  } else if (chat_service_impl_) {
    chat_completions_handler_ = [this](ClosureGuard& guard,
                                       brpc::Controller* ctrl,
                                       const proto::HttpRequest* request,
                                       proto::HttpResponse* response) {
      chat_completions_http_impl<ChatCall, ChatServiceImpl>(
          chat_service_impl_,
          guard,
          ctrl,
          request,
          response,
          ChatJsonParser::get(ServingMode::LLM));
    };
  }
}

void APIService::ChatCompletions(::google::protobuf::RpcController* controller,
                                 const proto::ChatRequest* request,
                                 proto::ChatResponse* response,
                                 ::google::protobuf::Closure* done) {
  // TODO with xllm-service
  xllm::ClosureGuard done_guard(
      done,
      std::bind(request_in_metric, nullptr),
      std::bind(request_out_metric, (void*)controller));
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  // Maybe need double check later

  chat_service_impl_->process_async_rpc_impl(request);
}

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

  if (!chat_completions_handler_) {
    LOG(ERROR) << "No chat completions handler registered";
    return;
  }

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  chat_completions_handler_(done_guard, ctrl, request, response);
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
  if (embedding_service_impl_) {
    handle_embedding_request<EmbeddingCall, EmbeddingServiceImpl>(
        embedding_service_impl_, controller, request, response, done);
  } else if (mm_embedding_service_impl_) {
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
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  bool st_models = models_service_impl_->list_models(nullptr, response);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (!st_models) {
    ctrl->SetFailed("list models failed.");
    LOG(ERROR) << "list models failed.";
    return;
  }
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
      google::protobuf::Arena::CreateMessage<proto::ModelListResponse>(arena);

  bool st_models = models_service_impl_->list_models(nullptr, resp_pb);
  if (!st_models) {
    LOG(ERROR) << "list models failed.";
    return;
  }

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

  auto [preprocess_status, processed_json] =
      ChatJsonParser::anthropic().preprocess(std::move(attachment));
  if (!preprocess_status.ok()) {
    ctrl->SetFailed(preprocess_status.message());
    LOG(ERROR) << "Anthropic JSON preprocessing failed: "
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

  if (anthropic_service_impl_) {
    handle_anthropic_messages(
        anthropic_service_impl_, done_guard, ctrl, request, response);
  } else {
    ctrl->SetFailed("Anthropic messages API is only supported for LLM engine");
    LOG(ERROR) << "Anthropic messages API is only supported for LLM engine";
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
  options.master_status() = MasterStatus(request->master_status());

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
                            proto::Status* response,
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
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

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

  if (to_serving_mode(master_->engine_type()) != ServingMode::LLM) {
    LOG(ERROR) << "fork master only supports LLM engine";
    return;
  }

  Options master_options;
  if (!ParseForkMasterRequest(req_pb, master_options)) {
    LOG(ERROR) << "Failed to parse fork master request";
    return;
  }

  if (has_model_master(master_options.model_id())) {
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
  if (master->is_sleeping() &&
      !master->get_rate_limiter()->try_set_sleeping()) {
    // Notice: this branch is only entered in exceptional cases.
    int32_t num_requests =
        master->get_rate_limiter()->get_num_concurrent_requests();
    LOG(FATAL) << "Cannot sleep model " << req_pb->model_id() << " with "
               << num_requests << " in-flight requests";
    ctrl->SetFailed("Cannot sleep model with in-flight requests");
    return;
  }

  if (!add_model_master_if_absent(master_options.model_id(), master.get())) {
    LOG(INFO) << "Master for model " << master_options.model_id()
              << " already exists";
    return;
  }
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
                       proto::Status* response,
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
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

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

  const auto req_master_status = MasterStatus(req_pb->master_status());
  if (req_master_status != MasterStatus::LIGHT_SLEEP &&
      req_master_status != MasterStatus::DEEP_SLEEP) {
    LOG(ERROR) << "Invalid sleep status: " << req_pb->master_status();
    ctrl->SetFailed("Invalid sleep status");
    return;
  }

  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  if (master->is_sleeping()) {
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
  master->set_master_status(req_master_status);
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
                        proto::Status* response,
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
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

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
  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  if (!master->is_sleeping()) {
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
          segments.emplace_back(proto_seg.offset(), proto_seg.size());
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

  master->set_master_status(MasterStatus::WAKEUP);
  // Success: return HTTP 200 with empty body
}

void APIService::LinkD2D(::google::protobuf::RpcController* controller,
                         const proto::D2DLinkRequest* request,
                         proto::Status* response,
                         ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  Master* master = get_model_master(request->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_ok(false);
    return;
  }
  bool status = master->link_d2d(
      {request->device_ips().begin(), request->device_ips().end()});
  response->set_ok(status);
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
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

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

  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  bool status = master->link_d2d(
      {req_pb->device_ips().begin(), req_pb->device_ips().end()});
  resp_pb->set_ok(status);

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
                           proto::Status* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  Master* master = get_model_master(request->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << request->model_id() << " not found";
    response->set_ok(false);
    return;
  }
  bool status = master->unlink_d2d(
      {request->device_ips().begin(), request->device_ips().end()});
  response->set_ok(status);
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
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

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

  Master* master = get_model_master(req_pb->model_id());
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }
  bool status = master->unlink_d2d(
      {req_pb->device_ips().begin(), req_pb->device_ips().end()});
  resp_pb->set_ok(status);

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
