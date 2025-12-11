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

#include "rerank_service_impl.h"

#include <torch/torch.h>

#include <string>

#include "common/instance_name.h"
#include "distributed_runtime/llm_master.h"
#include "framework/request/request_params.h"
#include "util/blocking_counter.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {
RerankServiceImpl::RerankServiceImpl(LLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

// rerank_async for brpc
void RerankServiceImpl::process_async_impl(std::shared_ptr<RerankCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  std::vector<std::string> documents;
  if (rpc_request.documents_size() > 0) {
    documents = std::vector<std::string>(rpc_request.documents().begin(),
                                         rpc_request.documents().end());
  }
  documents.emplace_back(rpc_request.query());

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

  master_->handle_batch_request(documents, sps, batch_callback);

  // Wait for all tasks to complete
  counter.wait();

  // calculate cosine similarity to get score
  size_t doc_size = documents.size() - 1;
  std::string query = documents[doc_size];
  auto query_embed = req_outputs[doc_size].outputs[0].embeddings.value();
  auto query_tensor = torch::from_blob(
      query_embed.data(), {query_embed.size()}, torch::kFloat32);

  std::vector<RerankRequestOutput> rerank_outputs;
  rerank_outputs.reserve(doc_size);
  for (size_t i = 0; i < doc_size; ++i) {
    if (req_outputs[i].outputs[0].embeddings.has_value()) {
      auto doc_embed = req_outputs[i].outputs[0].embeddings.value();
      auto doc_tensor = torch::from_blob(
          doc_embed.data(), {doc_embed.size()}, torch::kFloat32);
      auto score =
          torch::cosine_similarity(query_tensor, doc_tensor, 0).item<float>();
      rerank_outputs.emplace_back(i, documents[i], score);
    }
  }

  // send result to client
  int32_t top_n = documents.size() - 1;
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

bool RerankServiceImpl::send_result_to_client_brpc(
    std::shared_ptr<RerankCall> call,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    int32_t top_n,
    std::vector<RerankRequestOutput>& rerank_outputs,
    const std::vector<RequestOutput>& req_outputs) {
  auto& response = call->response();
  response.set_id(request_id);
  response.set_model(model);

  std::sort(rerank_outputs.begin(),
            rerank_outputs.end(),
            [](const RerankRequestOutput& a, const RerankRequestOutput& b) {
              return a.score > b.score;
            });

  // add top_n results
  response.mutable_results()->Reserve(top_n);
  for (int32_t i = 0; i < top_n; ++i) {
    auto* result = response.add_results();
    result->set_index(rerank_outputs[i].index);
    auto* document = result->mutable_document();
    document->set_text(rerank_outputs[i].document);
    result->set_relevance_score(rerank_outputs[i].score);
  }

  // add usage statistics
  int32_t num_prompt_tokens = 0;
  int32_t num_generated_tokens = 0;
  int32_t num_total_tokens = 0;
  for (auto req_output : req_outputs) {
    if (req_output.usage.has_value()) {
      const auto& usage = req_output.usage.value();
      num_prompt_tokens += usage.num_prompt_tokens;
      num_generated_tokens += usage.num_generated_tokens;
      num_total_tokens += usage.num_total_tokens;
    }
  }
  if (num_total_tokens > 0) {
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(num_prompt_tokens);
    proto_usage->set_completion_tokens(num_generated_tokens);
    proto_usage->set_total_tokens(num_total_tokens);
  }

  return call->write_and_finish(response);
}

}  // namespace xllm
