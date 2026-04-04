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

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/global_flags.h"
#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/framework/request/request_output.h"

#ifdef likely
#undef likely
#endif
#define likely(x) __builtin_expect(!!(x), 1)

#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {
void append_rec_scores(proto::Content* scores_context,
                       const SequenceOutput& output,
                       int32_t expected_count) {
  const auto& token_scores = output.token_ids_logprobs;
  const int32_t actual_count = static_cast<int32_t>(token_scores.size());
  CHECK(actual_count == 0 || actual_count == expected_count)
      << "REC sku_logprobs width mismatch, actual_count=" << actual_count
      << ", expected_count=" << expected_count
      << ", token_ids=" << output.token_ids.size();

  for (int32_t i = 0; i < expected_count; ++i) {
    if (i < actual_count && token_scores[i].has_value()) {
      scores_context->mutable_fp32_contents()->Add(token_scores[i].value());
    } else {
      scores_context->mutable_fp32_contents()->Add(0.0f);
    }
  }
}

std::vector<int64_t> select_rec_item_ids(const SequenceOutput& output) {
  if (!FLAGS_enable_rec_multi_item_output || output.item_ids_list.empty()) {
    if (output.item_ids.has_value()) {
      return {output.item_ids.value()};
    }
    return {};
  }

  std::vector<int64_t> selected_item_ids;
  selected_item_ids.reserve(output.item_ids_list.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const int64_t item_id : output.item_ids_list) {
    if (seen_item_ids.insert(item_id).second) {
      selected_item_ids.push_back(item_id);
    }
  }

  const int32_t each_threshold = FLAGS_each_conversion_threshold;
  if (each_threshold > 0 &&
      static_cast<int32_t>(selected_item_ids.size()) > each_threshold) {
    uint32_t seed = FLAGS_random_seed >= 0
                        ? static_cast<uint32_t>(FLAGS_random_seed) +
                              static_cast<uint32_t>(output.index)
                        : std::random_device{}();
    std::mt19937 generator(seed);
    std::shuffle(
        selected_item_ids.begin(), selected_item_ids.end(), generator);
    selected_item_ids.resize(each_threshold);
  }

  return selected_item_ids;
}

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
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
  }

  // Add rec specific output tensors
  auto output_tensor = response.mutable_output_tensors()->Add();
  output_tensor->set_name("rec_result");
  proto::InferOutputTensor* scores_tensor = nullptr;
  int32_t score_width = 0;
  if (FLAGS_enable_rec_score_output) {
    scores_tensor = response.mutable_output_tensors()->Add();
    scores_tensor->set_name("sku_logprobs");
    scores_tensor->set_datatype(proto::DataType::FLOAT);
    score_width = req_output.outputs.empty()
                      ? 0
                      : static_cast<int32_t>(
                            req_output.outputs.front().token_ids_logprobs.size());
  }

  if (FLAGS_enable_convert_tokens_to_item) {
    output_tensor->set_datatype(proto::DataType::INT64);
    std::vector<std::vector<int64_t>> selected_item_groups;
    selected_item_groups.reserve(req_output.outputs.size());

    int32_t total_item_count = 0;
    const int32_t total_threshold = FLAGS_total_conversion_threshold;
    for (const auto& output : req_output.outputs) {
      std::vector<int64_t> selected_item_ids = select_rec_item_ids(output);
      if (total_threshold > 0 &&
          total_item_count + static_cast<int32_t>(selected_item_ids.size()) >
              total_threshold) {
        const int32_t remaining_count =
            std::max(total_threshold - total_item_count, 0);
        if (remaining_count == 0) {
          selected_item_ids.clear();
        } else {
          selected_item_ids.resize(remaining_count);
        }
      }
      total_item_count += static_cast<int32_t>(selected_item_ids.size());
      selected_item_groups.push_back(std::move(selected_item_ids));
    }

    output_tensor->mutable_shape()->Add(total_item_count);
    if (scores_tensor != nullptr) {
      scores_tensor->mutable_shape()->Add(total_item_count);
      scores_tensor->mutable_shape()->Add(score_width);
    }

    auto* output_context = output_tensor->mutable_contents();
    auto* scores_context =
        scores_tensor == nullptr ? nullptr : scores_tensor->mutable_contents();
    for (int i = 0; i < req_output.outputs.size(); ++i) {
      const auto& selected_item_ids = selected_item_groups[i];
      for (const int64_t item_id : selected_item_ids) {
        output_context->mutable_int64_contents()->Add(item_id);
        if (scores_context != nullptr) {
          append_rec_scores(scores_context, req_output.outputs[i], score_width);
        }
      }
    }
  } else {
    output_tensor->set_datatype(proto::DataType::INT32);

    if (req_output.outputs.empty()) {
      output_tensor->mutable_shape()->Add(0);
      output_tensor->mutable_shape()->Add(0);
      if (scores_tensor != nullptr) {
        scores_tensor->mutable_shape()->Add(0);
        scores_tensor->mutable_shape()->Add(0);
      }
      return call->write_and_finish(response);
    }

    output_tensor->mutable_shape()->Add(req_output.outputs.size());
    output_tensor->mutable_shape()->Add(req_output.outputs[0].token_ids.size());
    if (scores_tensor != nullptr) {
      scores_tensor->mutable_shape()->Add(req_output.outputs.size());
      scores_tensor->mutable_shape()->Add(score_width);
    }

    auto context = output_tensor->mutable_contents();
    auto* scores_context =
        scores_tensor == nullptr ? nullptr : scores_tensor->mutable_contents();
    for (int i = 0; i < req_output.outputs.size(); ++i) {
      // LOG(INFO) << req_output.outputs[i].token_ids;
      context->mutable_int_contents()->Add(
          req_output.outputs[i].token_ids.begin(),
          req_output.outputs[i].token_ids.end());
      if (scores_context != nullptr) {
        append_rec_scores(scores_context, req_output.outputs[i], score_width);
      }
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
  if (FLAGS_enable_rec_score_output) {
    request_params.logprobs = true;
  }
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
