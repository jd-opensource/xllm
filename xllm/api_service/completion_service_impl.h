/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "api_service_impl.h"
#include "completion.pb.h"
#include "stream_call.h"

namespace xllm {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  CompletionServiceImpl(LLMMaster* master,
                        const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(CompletionServiceImpl);
};

}  // namespace xllm
