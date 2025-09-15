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

#include "image_generation_service_impl.h"

#include <glog/logging.h>

#include <string>

#include "common/instance_name.h"
#include "framework/request/dit_request_output.h"
#include "framework/request/dit_request_params.h"
#include "runtime/dit_master.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {
namespace {

bool send_result_to_client_brpc(std::shared_ptr<ImageGenerationCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const DiTRequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("list");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);
  auto* proto_output = response.mutable_output();
  const std::vector<DiTGenerationOutput>& outputs = req_output.outputs;
  proto_output->mutable_results()->Reserve(outputs.size());
  for (const auto& output : outputs) {
    auto* proto_result = proto_output->add_results();
    // proto_result->set_base64(output.image_tensor); // TODO proto tensor to
    // base64
    proto_result->set_width(output.width);
    proto_result->set_height(output.height);
    proto_result->set_seed(output.seed);
  }
  return call->write_and_finish(response);
}

}  // namespace

ImageGenerationServiceImpl::ImageGenerationServiceImpl(
    DiTMaster* master,
    const std::vector<std::string>& models)
    : master_{master}, models_{models.begin(), models.end()} {
  CHECK(master != nullptr);
  CHECK(!models_.empty());
}

// image_generation_async for brpc
void ImageGenerationServiceImpl::process_async(
    std::shared_ptr<ImageGenerationCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }
  // create RequestParams for image generation request
  // set is_image_generation and max_tokens = 1 to control engine step once.
  DiTRequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  // TODO only support input_str for now
  auto& input = rpc_request.input().prompt();
  // schedule the request
  master_->handle_request(
      std::move(request_params),
      call.get(),
      [call,
       model,
       request_id = request_params.request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const DiTRequestOutput& req_output) -> bool {
        LOG(INFO) << "into callback before request finished";
        LOG(INFO) << req_output.outputs.size();
        LOG(INFO) << req_output.outputs[0].image_tensor;
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }
        LOG(INFO) << "into callback after request finished";
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
