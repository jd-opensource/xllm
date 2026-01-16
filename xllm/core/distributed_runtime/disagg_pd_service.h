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

#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_service_impl.h"

namespace xllm {

class DisaggPDService : public proto::DisaggPDService {
 public:
  explicit DisaggPDService(DisaggPDScheduler* scheduler, Engine* engine);
  explicit DisaggPDService() {}
  virtual ~DisaggPDService() = default;

  // for decode recv prefill request
  void AddNewRequests(::google::protobuf::RpcController* controller,
                      const proto::DisaggRequests* request,
                      proto::DisaggResponses* response,
                      ::google::protobuf::Closure* done) override;

  // for decode recv first token from prefill
  void FirstGeneration(::google::protobuf::RpcController* controller,
                       const proto::DisaggGenerationsRequests* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // for decode recv multiple tokens from prefill
  virtual void MultiGenerations(::google::protobuf::RpcController* controller,
                                const proto::DisaggGenerationsRequests* request,
                                proto::Status* response,
                                ::google::protobuf::Closure* done) override;

  virtual void SendPullSignal(::google::protobuf::RpcController* controller,
                              const proto::PullSignal* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) override;

  virtual void LinkInstance(::google::protobuf::RpcController* controller,
                            const proto::InstanceClusterInfo* request,
                            proto::Status* response,
                            ::google::protobuf::Closure* done) override;

  virtual void UnlinkInstance(::google::protobuf::RpcController* controller,
                              const proto::InstanceClusterInfo* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) override;

 protected:
  std::unique_ptr<DisaggPDServiceImpl> disagg_pd_service_impl_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DisaggPDService);
};

}  // namespace xllm
