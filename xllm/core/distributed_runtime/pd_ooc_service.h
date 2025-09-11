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
#include "pd_ooc_service_impl.h"

namespace xllm {

class PDOOCService : public proto::PDOOCService {
 public:
  explicit PDOOCService(PDOOCScheduler* scheduler, Engine* engine);
  virtual ~PDOOCService() = default;

  // for prefill recv decode response
  void Generation(::google::protobuf::RpcController* controller,
                  const proto::DisaggStreamGeneration* request,
                  proto::Status* response,
                  ::google::protobuf::Closure* done) override;

  // for prefill recv decode response
  void Generations(::google::protobuf::RpcController* controller,
                   const proto::DisaggStreamGenerations* requests,
                   proto::StatusSet* responses,
                   ::google::protobuf::Closure* done) override;

  // for decode recv prefill request
  void AddNewRequests(::google::protobuf::RpcController* controller,
                      const proto::DisaggRequests* request,
                      proto::DisaggResponses* response,
                      ::google::protobuf::Closure* done) override;

  // for decode recv first token from prefill
  void FirstGeneration(::google::protobuf::RpcController* controller,
                       const proto::DisaggGenerations* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PDOOCService);
  std::unique_ptr<PDOOCServiceImpl> pd_ooc_service_impl_;
};

}  // namespace xllm
