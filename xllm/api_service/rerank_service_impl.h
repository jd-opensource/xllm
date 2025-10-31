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
#include <absl/container/flat_hash_set.h>

#include "api_service/api_service_impl.h"
#include "api_service/call.h"
#include "api_service/non_stream_call.h"
#include "rerank.pb.h"

namespace xllm {

using RerankCall = NonStreamCall<proto::RerankRequest, proto::RerankResponse>;

struct RerankRequestOutput {
  int32_t index = 0;
  std::string document = "";
  float score = 0.0f;

  RerankRequestOutput(int32_t index, std::string document, float score)
      : index(index), document(std::move(document)), score(score) {}
};

// a class to handle completion requests
class RerankServiceImpl : public APIServiceImpl<RerankCall> {
 public:
  RerankServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<RerankCall> call);

 protected:
  bool send_result_to_client_brpc(
      std::shared_ptr<RerankCall> call,
      const std::string& request_id,
      int64_t created_time,
      const std::string& model,
      int32_t top_n,
      std::vector<RerankRequestOutput>& rerank_outputs,
      const std::vector<RequestOutput>& req_outputs);

 protected:
  DISALLOW_COPY_AND_ASSIGN(RerankServiceImpl);
  LLMMaster* master_ = nullptr;
};

}  // namespace xllm
