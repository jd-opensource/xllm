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

#include "disagg_pd_service_impl.h"

#include <glog/logging.h>

#include "common/types.h"
#include "distributed_runtime/llm_engine.h"
#include "framework/request/request_output.h"
#include "scheduler/disagg_pd_scheduler.h"

namespace xllm {

DisaggPDServiceImpl::DisaggPDServiceImpl(DisaggPDScheduler* scheduler,
                                         Engine* engine)
    : scheduler_(scheduler), engine_(engine) {}

std::shared_ptr<Request> DisaggPDServiceImpl::generate_request(
    const proto::DisaggRequest& req) {
  // create a new request
  // TODO: Should to support best_of > 1 case, now we only consider
  // to allocate blocks for the first sequence in the request.
  // But request maybe expend_sequence in running stage.
  std::string prompt = req.prompt();
  std::vector<int> prompt_tokens(req.prompt_tokens().begin(),
                                 req.prompt_tokens().end());

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = req.frequency_penalty();
  sampling_param.presence_penalty = req.presence_penalty();
  sampling_param.repetition_penalty = req.repetition_penalty();
  sampling_param.temperature = req.temperature();
  sampling_param.top_p = req.top_p();
  sampling_param.top_k = req.top_k();
  sampling_param.logprobs = req.logprobs();
  sampling_param.top_logprobs = req.top_logprobs();
  sampling_param.is_embeddings = req.is_embeddings();

  std::unordered_set<int32_t> stop_tokens;
  for (auto& stop_token_id : req.stop_token_ids()) {
    stop_tokens.insert(stop_token_id);
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  for (auto& stop_sequence : req.stop_sequences()) {
    auto stop_seq_tokens = std::vector<int32_t>(
        stop_sequence.seq_tokens().begin(), stop_sequence.seq_tokens().end());
    stop_sequences.push_back(std::move(stop_seq_tokens));
  }
  StoppingChecker stopping_checker(req.max_tokens(),
                                   req.max_context_len(),
                                   req.eos_token_id(),
                                   req.ignore_eos(),
                                   std::move(stop_tokens),
                                   std::move(stop_sequences));

  auto output_callback = [this](const RequestOutput& output) {
    return scheduler_->decode_send_stream_generation(output);
  };

  auto batch_output_callback =
      [this](const std::vector<RequestOutput>& outputs) {
        return scheduler_->decode_send_stream_generations(outputs);
      };

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         req.seq_capacity(),
                         req.n(),
                         req.best_of(),
                         req.logprobs(),
                         req.stream(),
                         req.echo(),
                         req.skip_special_tokens(),
                         scheduler_->enable_schedule_overlap(),
                         output_callback,
                         batch_output_callback);

  auto new_request = std::make_shared<Request>(
      req.req_id(),
      req.x_request_id(),
      req.x_request_time(),
      std::move(req_state),
      req.service_req_id(),
      req.offline(),
      req.slo_ms(),
      static_cast<xllm::RequestPriority>(req.priority()));

  // add one sequence, rest will be added by scheduler
  return new_request;
}

void DisaggPDServiceImpl::decode_recv_new_requests(
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response) {
  // link prefill instance
  if (!scheduler_->is_instance_linked(request->prefill_name())) {
    std::vector<uint64_t> cluster_ids(
        request->cluster_infos().cluster_ids().begin(),
        request->cluster_infos().cluster_ids().end());
    std::vector<std::string> addrs(request->cluster_infos().addrs().begin(),
                                   request->cluster_infos().addrs().end());
    std::vector<std::string> device_ips(
        request->cluster_infos().device_ips().begin(),
        request->cluster_infos().device_ips().end());
    std::vector<uint16_t> ports(request->cluster_infos().ports().begin(),
                                request->cluster_infos().ports().end());
    int32_t dp_size = request->cluster_infos().dp_size();
    if (!scheduler_->link_instance(request->prefill_name(),
                                   cluster_ids,
                                   addrs,
                                   device_ips,
                                   ports,
                                   dp_size)) {
      LOG(ERROR) << "Link instance failed, instance name : "
                 << request->prefill_name();
      return;
    }
  }

  for (auto& req : request->reqs()) {
    // Try to allocate blocks for new requests
    int32_t dp_rank = 0;
    auto blocks = scheduler_->allocate_raw_blocks(req.tokens_num(), dp_rank);
    auto resp = response->add_resps();
    if (blocks.empty()) {
      resp->set_req_id(req.req_id());
      // FIXME: set status code
      resp->set_status_code(404);
    } else {
      resp->set_req_id(req.req_id());

      auto new_request = generate_request(req);
      if (new_request == nullptr) {
        resp->set_status_code(500);
        continue;
      }

      resp->set_status_code(200);
      for (auto& block : blocks) {
        *(resp->mutable_blocks_ids()->Add()) = block.id();
      }
      resp->set_dp_rank(dp_rank);
      for (auto& sequence : new_request->sequences()) {
        sequence->set_dp_rank(dp_rank);
        sequence->add_kv_blocks(blocks);
        // prompt kv is computed in prefill instance,
        // so we update these received kvs(tokens_num) here.
        sequence->kv_state().incr_kv_cache_tokens_num(req.tokens_num());
      }

      // push the request to scheduler request buffer
      bool success =
          scheduler_->decode_schedule(new_request, request->prefill_name());
      if (!success) {
        LOG(ERROR) << "Failed to schedule new decode instance request: "
                   << req.req_id();
        // request and blocks are released in scheduler
        resp->set_status_code(500);
      }
    }
  }
}

// TODO: support embedding later, now we only support tokens
void DisaggPDServiceImpl::decode_recv_first_generation(
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response) {
  // TODO: we only support one request generation currently
  for (auto& gen : request->multi_gens()) {
    // Process the first token from the tokens array
    if (gen.tokens().empty()) {
      response->set_ok(false);
      return;
    }

    const auto& first_token = gen.tokens(0);
    std::vector<int64_t> top_tokens(first_token.top_tokens().begin(),
                                    first_token.top_tokens().end());
    std::vector<float> top_logprobs(first_token.top_logprobs().begin(),
                                    first_token.top_logprobs().end());
    std::vector<uint64_t> cluster_ids(gen.cluster_ids().begin(),
                                      gen.cluster_ids().end());
    std::vector<std::string> addrs(gen.addrs().begin(), gen.addrs().end());
    std::vector<int64_t> k_cache_ids(gen.k_cache_ids().begin(),
                                     gen.k_cache_ids().end());
    std::vector<int64_t> v_cache_ids(gen.v_cache_ids().begin(),
                                     gen.v_cache_ids().end());
    std::vector<uint64_t> block_ids(gen.block_ids().begin(),
                                    gen.block_ids().end());

    bool success =
        scheduler_->decode_recv_first_generation(gen.req_id(),
                                                 first_token.token_id(),
                                                 first_token.has_logprob(),
                                                 first_token.logprob(),
                                                 std::move(top_tokens),
                                                 std::move(top_logprobs),
                                                 gen.kv_cache_transfer_mode(),
                                                 std::move(cluster_ids),
                                                 std::move(addrs),
                                                 std::move(k_cache_ids),
                                                 std::move(v_cache_ids),
                                                 std::move(block_ids),
                                                 gen.dp_size(),
                                                 gen.dp_rank());
    if (!success) {
      response->set_ok(false);
      return;
    }
  }

  response->set_ok(true);
}

bool DisaggPDServiceImpl::prefill_recv_generation(
    const proto::DisaggStreamGeneration* request,
    proto::Status* response) {
  // convert proto request to `RequestOutput`
  RequestOutput request_output;
  request_output.request_id = request->req_id();
  request_output.service_request_id = request->service_req_id();
  if (request->has_gen_status()) {
    request_output.status =
        Status(static_cast<StatusCode>(request->gen_status().status_code()),
               request->gen_status().status_msg());
  }
  if (request->has_usage()) {
    Usage u;
    u.num_prompt_tokens = request->usage().num_prompt_tokens();
    u.num_generated_tokens = request->usage().num_generated_tokens();
    u.num_total_tokens = request->usage().num_total_tokens();
    request_output.usage = std::move(u);
  }
  request_output.finished = request->finished();
  for (auto& output : request->outputs()) {
    SequenceOutput sequence_output;
    sequence_output.index = output.index();
    sequence_output.text = output.text();
    sequence_output.token_ids = std::vector<int32_t>(output.token_ids().begin(),
                                                     output.token_ids().end());
    if (!output.finish_reason().empty()) {
      sequence_output.finish_reason = output.finish_reason();
    }
    if (output.logprobs().size() > 0) {
      std::vector<LogProb> logprobs;
      for (auto& logprob : output.logprobs()) {
        LogProb lp;
        lp.token = logprob.log_prob_data().token();
        lp.token_id = logprob.log_prob_data().token_id();
        lp.logprob = logprob.log_prob_data().logprob();
        lp.finished_token = logprob.log_prob_data().finished_token();
        if (logprob.top_logprobs().size() > 0) {
          std::vector<LogProbData> top_logprobs;
          for (auto& top_logprob : logprob.top_logprobs()) {
            LogProbData lpd;
            lpd.token = top_logprob.token();
            lpd.token_id = top_logprob.token_id();
            lpd.logprob = top_logprob.logprob();
            lpd.finished_token = top_logprob.finished_token();
            top_logprobs.emplace_back(std::move(lpd));
          }
          lp.top_logprobs = std::move(top_logprobs);
        }
        logprobs.emplace_back(std::move(lp));
      }
      sequence_output.logprobs = std::move(logprobs);
    }
    request_output.outputs.emplace_back(std::move(sequence_output));
  }

  // TODO: handle error later
  bool success = scheduler_->prefill_recv_generation(request_output);

  // we don't set response in batch_responses case.
  // it will be set in function `prefill_recv_generations`
  if (response) {
    response->set_ok(success);
    // TODO: handle error later
  }

  return success;
}

void DisaggPDServiceImpl::prefill_recv_generations(
    const proto::DisaggStreamGenerations* requests,
    proto::StatusSet* responses) {
  for (auto& gen : requests->gens()) {
    responses->mutable_all_status()->Add()->set_ok(
        prefill_recv_generation(&gen, nullptr));
  }
}

}  // namespace xllm
