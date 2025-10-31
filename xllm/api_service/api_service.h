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

#pragma once

#include "chat_service_impl.h"
#include "completion_service_impl.h"
#include "embedding_service_impl.h"
#include "image_generation_service_impl.h"
#include "models_service_impl.h"
#include "qwen3_rerank_service_impl.h"
#include "rerank_service_impl.h"
#include "xllm_service.pb.h"

namespace xllm {

class APIService : public proto::XllmAPIService {
 public:
  APIService(Master* master,
             const std::vector<std::string>& model_names,
             const std::vector<std::string>& model_versions);
  ~APIService() = default;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::CompletionRequest* request,
                   proto::CompletionResponse* response,
                   ::google::protobuf::Closure* done) override;

  void CompletionsHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::ChatRequest* request,
                       proto::ChatResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletionsHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::EmbeddingRequest* request,
                  proto::EmbeddingResponse* response,
                  ::google::protobuf::Closure* done) override;

  void EmbeddingsHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void ImageGeneration(::google::protobuf::RpcController* controller,
                       const proto::ImageGenerationRequest* request,
                       proto::ImageGenerationResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ImageGenerationHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Rerank(::google::protobuf::RpcController* controller,
              const proto::RerankRequest* request,
              proto::RerankResponse* response,
              ::google::protobuf::Closure* done) override;

  void RerankHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::ModelListRequest* request,
              proto::ModelListResponse* response,
              ::google::protobuf::Closure* done) override;

  void ModelsHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void GetCacheInfo(::google::protobuf::RpcController* controller,
                    const proto::HttpRequest* request,
                    proto::HttpResponse* response,
                    ::google::protobuf::Closure* done) override;

  void LinkCluster(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void UnlinkCluster(::google::protobuf::RpcController* controller,
                     const proto::HttpRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done) override;

  void ModelVersionsHttp(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done) override;

 private:
  Master* master_;

  std::unique_ptr<CompletionServiceImpl> completion_service_impl_;
  std::unique_ptr<ChatServiceImpl> chat_service_impl_;
  std::unique_ptr<MMChatServiceImpl> mm_chat_service_impl_;
  std::unique_ptr<EmbeddingServiceImpl> embedding_service_impl_;
  std::unique_ptr<ModelsServiceImpl> models_service_impl_;
  std::unique_ptr<ImageGenerationServiceImpl> image_generation_service_impl_;
  std::unique_ptr<RerankServiceImpl> rerank_service_impl_;
  std::unique_ptr<RecCompletionServiceImpl> rec_completion_service_impl_;
};

}  // namespace xllm
