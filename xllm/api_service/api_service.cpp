#include "api_service.h"

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include "call.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/metrics.h"
#include "core/runtime/llm_master.h"
#include "core/runtime/vlm_master.h"
#include "core/util/closure_guard.h"
#include "embedding.pb.h"
#include "models.pb.h"
#include "service_impl_factory.h"
#include "xllm_metrics.h"
#include <google/protobuf/util/json_util.h>
namespace xllm {

APIService::APIService(Master* master,
                       const std::vector<std::string>& model_names,
                       const std::vector<std::string>& model_versions)
    : master_(master) {
  if (FLAGS_backend == "llm") {
    auto llm_master = dynamic_cast<LLMMaster*>(master);
    completion_service_impl_ =
        ServiceImplFactory<CompletionServiceImpl>::create_service_impl(
            llm_master, model_names);
    chat_service_impl_ =
        ServiceImplFactory<ChatServiceImpl>::create_service_impl(llm_master,
                                                                 model_names);
    embedding_service_impl_ =
        ServiceImplFactory<EmbeddingServiceImpl>::create_service_impl(
            llm_master, model_names);
  } else if (FLAGS_backend == "vlm") {
    auto vlm_master = dynamic_cast<VLMMaster*>(master);
    mm_chat_service_impl_ =
        std::make_unique<MMChatServiceImpl>(vlm_master, model_names);
  }
  models_service_impl_ =
      ServiceImplFactory<ModelsServiceImpl>::create_service_impl(
          model_names, model_versions);
}

void APIService::Completions(::google::protobuf::RpcController* controller,
                             const proto::CompletionRequest* request,
                             proto::CompletionResponse* response,
                             ::google::protobuf::Closure* done) {
  // TODO with xllm-service
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

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::CompletionResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string attachment = std::move(ctrl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
      ctrl, done_guard.release(), req_pb, resp_pb);
  completion_service_impl_->process_async(call);
}

void APIService::ChatCompletions(::google::protobuf::RpcController* controller,
                                 const proto::ChatRequest* request,
                                 proto::ChatResponse* response,
                                 ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

namespace {
template <typename ChatCall, typename Service>
void ChatCompletionsImpl(std::unique_ptr<Service>& service,
                         xllm::ClosureGuard& guard,
                         ::google::protobuf::Arena* arena,
                         brpc::Controller* ctrl) {
  auto req_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ReqType>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<typename ChatCall::ResType>(arena);

  std::string attachment = std::move(ctrl->request_attachment().to_string());
  std::string error;

  auto json_status = google::protobuf::util::JsonStringToMessage(attachment, req_pb);
  if (!json_status.ok()) {
    ctrl->SetFailed(json_status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << json_status.ToString();
    return;
  }

  auto call =
      std::make_shared<ChatCall>(ctrl, guard.release(), req_pb, resp_pb);
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

  auto arena = response->GetArena();
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);

  if (FLAGS_backend == "llm") {
    CHECK(chat_service_impl_) << " chat service is invalid.";
    ChatCompletionsImpl<ChatCall, ChatServiceImpl>(
        chat_service_impl_, done_guard, arena, ctrl);
  } else if (FLAGS_backend == "vlm") {
    CHECK(mm_chat_service_impl_) << " mm chat service is invalid.";
    ChatCompletionsImpl<MMChatCall, MMChatServiceImpl>(
        mm_chat_service_impl_, done_guard, arena, ctrl);
  }
}

void APIService::Embeddings(::google::protobuf::RpcController* controller,
                            const proto::EmbeddingRequest* request,
                            proto::EmbeddingResponse* response,
                            ::google::protobuf::Closure* done) {
  // TODO with xllm-service
}

void APIService::EmbeddingsHttp(::google::protobuf::RpcController* controller,
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

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::EmbeddingRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::EmbeddingResponse>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string attachment = std::move(ctrl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
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
      ctrl, done_guard.release(), req_pb, resp_pb);
  embedding_service_impl_->process_async(call);
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

void APIService::GetCacheInfo(::google::protobuf::RpcController* controller,
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
      google::protobuf::Arena::CreateMessage<proto::GetCacheInfoResponse>(
          arena);

  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  std::vector<int64_t> k_cache_ids;
  std::vector<int64_t> v_cache_ids;
  master_->get_cache_info(cluster_ids, addrs, k_cache_ids, v_cache_ids);

  resp_pb->mutable_cluster_ids()->Add(cluster_ids.begin(), cluster_ids.end());
  resp_pb->mutable_addrs()->Add(addrs.begin(), addrs.end());
  resp_pb->mutable_k_cache_ids()->Add(k_cache_ids.begin(), k_cache_ids.end());
  resp_pb->mutable_v_cache_ids()->Add(v_cache_ids.begin(), v_cache_ids.end());

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(*resp_pb, &json_output, &err_msg)) {
    LOG(ERROR) << "proto to json failed";
    return;
  }
}

void APIService::LinkCluster(::google::protobuf::RpcController* controller,
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
      google::protobuf::Arena::CreateMessage<proto::ClusterInfos>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string attachment = std::move(ctrl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::vector<uint64_t> cluster_ids(req_pb->cluster_ids().begin(),
                                    req_pb->cluster_ids().end());
  std::vector<std::string> addrs;
  addrs.reserve(req_pb->addrs_size());
  for (int i = 0; i < req_pb->addrs_size(); ++i) {
    addrs.emplace_back(std::move(*req_pb->mutable_addrs()->Mutable(i)));
  }
  std::vector<std::string> device_ips;
  device_ips.reserve(req_pb->device_ips_size());
  for (int i = 0; i < req_pb->device_ips_size(); ++i) {
    device_ips.emplace_back(
        std::move(*req_pb->mutable_device_ips()->Mutable(i)));
  }
  std::vector<uint16_t> ports(req_pb->ports().begin(), req_pb->ports().end());

  bool status = master_->link_cluster(
      cluster_ids, addrs, device_ips, ports, req_pb->dp_size());

  resp_pb->set_status(status);

  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(*resp_pb, &json_output, &err_msg)) {
    LOG(ERROR) << "proto to json failed";
    return;
  }
}

void APIService::UnlinkCluster(::google::protobuf::RpcController* controller,
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
      google::protobuf::Arena::CreateMessage<proto::ClusterInfos>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<proto::RpcStatus>(arena);

  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  std::string attachment = std::move(ctrl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::vector<uint64_t> cluster_ids(req_pb->cluster_ids().begin(),
                                    req_pb->cluster_ids().end());
  std::vector<std::string> addrs;
  addrs.reserve(req_pb->addrs_size());
  for (int i = 0; i < req_pb->addrs_size(); ++i) {
    addrs.emplace_back(std::move(*req_pb->mutable_addrs()->Mutable(i)));
  }
  std::vector<std::string> device_ips;
  device_ips.reserve(req_pb->device_ips_size());
  for (int i = 0; i < req_pb->device_ips_size(); ++i) {
    device_ips.emplace_back(
        std::move(*req_pb->mutable_device_ips()->Mutable(i)));
  }
  std::vector<uint16_t> ports(req_pb->ports().begin(), req_pb->ports().end());

  bool status = master_->unlink_cluster(
      cluster_ids, addrs, device_ips, ports, req_pb->dp_size());

  resp_pb->set_status(status);

  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(*resp_pb, &json_output, &err_msg)) {
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

}  // namespace xllm
