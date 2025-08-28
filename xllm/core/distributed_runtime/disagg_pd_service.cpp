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

#include "disagg_pd_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

namespace xllm {

DisaggPDService::DisaggPDService(DisaggPDScheduler* scheduler, Engine* engine) {
  disagg_pd_service_impl_ =
      std::make_unique<DisaggPDServiceImpl>(scheduler, engine);
}

void DisaggPDService::Generation(::google::protobuf::RpcController* controller,
                                 const proto::DisaggStreamGeneration* request,
                                 proto::Status* response,
                                 ::google::protobuf::Closure* done) {
  // receive generations from Decode
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->prefill_recv_generation(request, response);
}

void DisaggPDService::Generations(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggStreamGenerations* requests,
    proto::StatusSet* responses,
    ::google::protobuf::Closure* done) {
  // receive generations from Decode
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->prefill_recv_generations(requests, responses);
}

void DisaggPDService::AddNewRequests(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  // try to allocate blocks for new requests
  disagg_pd_service_impl_->decode_recv_new_requests(request, response);
}

// TODO: support embedding later, now we only support tokens
void DisaggPDService::FirstGeneration(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggGenerations* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  // Receive first token from Prefill, schedule the request to running queue
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->decode_recv_first_generation(request, response);
}

}  // namespace xllm
