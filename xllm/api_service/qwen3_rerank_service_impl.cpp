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

#include "api_service/qwen3_rerank_service_impl.h"

#include "util/blocking_counter.h"

namespace xllm {

Qwen3RerankServiceImpl::Qwen3RerankServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : RerankServiceImpl(master, models) {}

void Qwen3RerankServiceImpl::process_async_impl(
    std::shared_ptr<RerankCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  auto query = rpc_request.query();
  std::vector<std::string> documents;
  if (rpc_request.documents_size() > 0) {
    documents = std::vector<std::string>(rpc_request.documents().begin(),
                                         rpc_request.documents().end());
  }
  std::vector<std::string> reqs;
  reqs.reserve(documents.size());
  for (size_t i = 0; i < documents.size(); ++i) {
    reqs.emplace_back(query + documents[i]);
  }

  // create RequestParams for rerank request
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  std::vector<RequestParams> sps(documents.size(), request_params);
  auto request_id = request_params.request_id;
  auto created_time = absl::ToUnixSeconds(absl::Now());

  // schedule the request
  std::vector<RequestOutput> req_outputs;
  req_outputs.resize(documents.size());
  BlockingCounter counter(documents.size());

  auto batch_callback = [&req_outputs, &counter](size_t index,
                                                 RequestOutput output) -> bool {
    req_outputs[index] = std::move(output);
    counter.decrement_count();
    return true;
  };

  master_->handle_batch_request(reqs, sps, batch_callback);

  // Wait for all tasks to complete
  counter.wait();

  // get score
  size_t doc_size = documents.size();
  std::vector<RerankRequestOutput> rerank_outputs;
  rerank_outputs.reserve(doc_size);
  for (size_t i = 0; i < doc_size; ++i) {
    if (req_outputs[i].outputs[0].logprobs.has_value()) {
      auto score = req_outputs[i].outputs[0].logprobs.value()[0].logprob;
      rerank_outputs.emplace_back(i, documents[i], score);
    }
  }

  // send result to client
  int32_t top_n = documents.size();
  if (rpc_request.has_top_n()) {
    top_n = std::min(top_n, rpc_request.top_n());
  }
  send_result_to_client_brpc(call,
                             request_id,
                             created_time,
                             model,
                             top_n,
                             rerank_outputs,
                             req_outputs);
}

}  // namespace xllm