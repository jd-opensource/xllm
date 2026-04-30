/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "api_service/api_service_impl.h"
#include "api_service/non_stream_call.h"
#include "core/distributed_runtime/vlm_master.h"
#include "speech.pb.h"

namespace xllm {

using SpeechCall = NonStreamCall<proto::SpeechRequest, proto::SpeechResponse>;

class SpeechServiceImpl final : public APIServiceImpl<SpeechCall> {
 public:
  SpeechServiceImpl(VLMMaster* master, const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<SpeechCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(SpeechServiceImpl);

  VLMMaster* master_ = nullptr;
  std::string default_model_;
};

}  // namespace xllm
