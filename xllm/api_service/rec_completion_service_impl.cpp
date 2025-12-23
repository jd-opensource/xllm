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

#include "rec_completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

#include "common/global_flags.h"
#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/framework/request/request_output.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {
void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

bool send_result_to_client_brpc_rec(std::shared_ptr<CompletionCall> call,
                                    const std::string& request_id,
                                    int64_t created_time,
                                    const std::string& model,
                                    const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  // add choices into response
  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  // Add rec specific output tensors
  auto output_tensor = response.mutable_output_tensors()->Add();
  output_tensor->set_name("rec_result");
  if (FLAGS_enable_constrained_decoding) {
    output_tensor->set_datatype(proto::DataType::INT64);
    output_tensor->mutable_shape()->Add(req_output.outputs.size());
    output_tensor->mutable_shape()->Add(1);  // Single item per output
    // TODO: add following when next pr.
    /*
    auto context = output_tensor->mutable_contents();
    for (int i = 0; i < req_output.outputs.size(); ++i) {
      if (req_output.outputs[i].item_ids.has_value()) {
        context->mutable_int64_contents()->Add(
            req_output.outputs[i].item_ids.value());
      }
    }
    */
  } else {
    output_tensor->set_datatype(proto::DataType::INT32);

    output_tensor->mutable_shape()->Add(req_output.outputs.size());
    output_tensor->mutable_shape()->Add(req_output.outputs[0].token_ids.size());

    auto context = output_tensor->mutable_contents();
    for (int i = 0; i < req_output.outputs.size(); ++i) {
      // LOG(INFO) << req_output.outputs[i].token_ids;
      context->mutable_int_contents()->Add(
          req_output.outputs[i].token_ids.begin(),
          req_output.outputs[i].token_ids.end());
    }
  }

  return call->write_and_finish(response);
}

}  // namespace

RecCompletionServiceImpl::RecCompletionServiceImpl(
    RecMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

void RecCompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  const auto& rpc_request = call->request();

  // check if model is supported
  const auto& model = rpc_request.model();
  if (unlikely(!models_.contains(model))) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited.
  if (unlikely(master_->get_rate_limiter()->is_limited())) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  const auto& rpc_request_ref = call->request();
  std::optional<std::vector<proto::InferInputTensor>> input_tensors =
      std::nullopt;
  if (rpc_request_ref.input_tensors_size()) {
    std::vector<proto::InferInputTensor> tensors;
    tensors.reserve(rpc_request_ref.input_tensors_size());
    for (int i = 0; i < rpc_request_ref.input_tensors_size(); ++i) {
      tensors.push_back(rpc_request_ref.input_tensors(i));
    }
    input_tensors = std::move(tensors);
  }

  // schedule the request
  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  master_->handle_request(
      std::move(rpc_request_ref.prompt()),
      std::move(prompt_tokens),
      std::move(input_tensors),
      std::move(request_params),
      [call,
       model,
       master = master_,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       request_id = saved_request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a request is
            // finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request is finished
        // or canceled.
        if (req_output.finished || req_output.cancelled) {
          master->get_rate_limiter()->decrease_one_request();
        }

        return send_result_to_client_brpc_rec(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
