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

#include "scheduler/disagg_pd_scheduler.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/server.h>

#include <random>

#include "common/global_flags.h"
#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "runtime/engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "util/env_var.h"

namespace xllm {

DisaggPDScheduler::DisaggPDScheduler(Engine* engine, const Options& options)
    : ContinuousScheduler(engine, options) {
  if (!options_.instance_role().has_value()) {
    LOG(FATAL) << "Instance type is not set in disagg pd mode.";
  }
  // start dispatch thread for prefill instance
  dispatch_thread_ = std::make_unique<std::thread>(
      &DisaggPDScheduler::dispatch_requests, this);

  rpc_server_thread_ =
      std::make_unique<std::thread>(&DisaggPDScheduler::start_rpc_server, this);
  // wait rpc server initialized
  auto rpc_server = ServerRegistry::get_instance().get_server("DisaggPDServer");
  while (!rpc_server || !rpc_server->has_initialized()) {
    absl::SleepFor(absl::Milliseconds(100));
    rpc_server = ServerRegistry::get_instance().get_server("DisaggPDServer");
  }
  // connect to master service
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
  xservice_client_->set_scheduler(this);

  // register instance info
  instance_info_.name = xservice_client_->get_instance_name();
  instance_info_.rpc_address = rpc_server->listen_address();
  instance_info_.type = options_.instance_role().value().to_string();
  LOG(INFO) << "Instance info: instance name = " << instance_info_.name
            << ", instance rpc_address = " << instance_info_.rpc_address
            << ", instance type = " << instance_info_.type;

  engine->get_cache_info(instance_info_.cluster_ids,
                         instance_info_.addrs,
                         instance_info_.k_cache_ids,
                         instance_info_.v_cache_ids);
  instance_info_.dp_size = options.dp_size();

  // profile ttft and update instance info
  profile_ttft();
}

DisaggPDScheduler::~DisaggPDScheduler() {
  if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
    rpc_server_thread_->join();
  }

  if (dispatch_thread_ && dispatch_thread_->joinable()) {
    dispatch_thread_->join();
  }
}

void DisaggPDScheduler::profile_ttft() {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }
  int32_t vocab_size = model_args.vocab_size();

  // warm up
  run_prefill_request(max_context_len, vocab_size);

  // get TTFT starting from max_context_len, dividing the token length by 2 in
  // each loop iteration
  for (int32_t token_length = max_context_len; token_length > 1;
       token_length >>= 1) {
    int64_t latency = run_prefill_request(token_length, vocab_size);
    instance_info_.ttft_profiling_data.emplace_back(
        std::make_pair(token_length, latency));
  }
}

int64_t DisaggPDScheduler::run_prefill_request(int32_t token_length,
                                               int32_t vocab_size) {
  // generate random token ids and request
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // generate token id within the range [0, vocab_size - 1]
  std::uniform_int_distribution<int32_t> dis(0, vocab_size - 1);
  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() { return dis(gen); });

  // generate request
  RequestState req_state(token_ids);
  Request request(/*request_id=*/"",
                  /*x_request_id=*/"",
                  /*x_request_time=*/"",
                  req_state);

  // allocate blocks
  if (!kv_cache_manager_client_->allocate(request.sequences()[0].get())) {
    LOG(FATAL) << "Profiling TTFT failed! Not enough blocks, token length : "
               << token_length;
  }

  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(
              {request.sequences()[0].get()},
              {token_length},
              kv_cache_manager_client_->get_copy_in_cache_block_infos(),
              kv_cache_manager_client_->get_copy_out_cache_block_infos());

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  const int64_t latency = absl::ToInt64Milliseconds(absl::Now() - start_time);
  kv_cache_manager_client_->deallocate(&request);

  return latency;
}

// TODO: maybe we should consider update info case even if info already exists
// in local.
bool DisaggPDScheduler::check_remote_instance_info(
    const std::string& instance_name) {
  if (remote_instances_info_.find(instance_name) !=
      remote_instances_info_.end()) {
    return true;
  }

  InstanceInfo instance_info =
      xservice_client_->get_instance_info(instance_name);
  if (instance_info.name.empty()) {
    LOG(ERROR)
        << "Failed to get instance info from master server, instance name: "
        << instance_name;
    return false;
  }

  remote_instances_info_[instance_name] = instance_info;
  return true;
}

proto::DisaggPDService_Stub* DisaggPDScheduler::create_rpc_channel(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(instance_channel_map_mutex_);
  auto it = instance_channel_map_.find(instance_name);
  if (it == instance_channel_map_.end()) {
    LOG(WARNING) << "Failed to find channel to instance: " << instance_name
                 << ", try to create channel now.";
    // check prefill instance info
    if (!check_remote_instance_info(instance_name)) {
      LOG(ERROR) << "Check remote instance info failed, instance name: "
                 << instance_name;
      return nullptr;
    }
    // create channel to prefill instance
    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = FLAGS_timeout_ms;
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(remote_instances_info_[instance_name].rpc_address.c_str(),
                      load_balancer.c_str(),
                      &options) != 0) {
      LOG(ERROR) << "Fail to initialize channel for "
                 << remote_instances_info_[instance_name].rpc_address;
      remote_instances_info_.erase(instance_name);
      delete channel;
      return nullptr;
    }
    // req_to_channel_map_[request.request_id] = channel;
    proto::DisaggPDService_Stub* stub =
        new proto::DisaggPDService_Stub(channel);
    instance_channel_map_[instance_name] = stub;
    return stub;
  }

  return it->second;
}

void DisaggPDScheduler::start_rpc_server() {
  std::unique_ptr<DisaggPDService> service =
      std::make_unique<DisaggPDService>(this, engine_);
  auto rpc_server =
      ServerRegistry::get_instance().register_server("DisaggPDServer");
  if (!rpc_server->start(std::move(service))) {
    LOG(ERROR) << "Failed to start brpc disagg pd server on port "
               << FLAGS_disagg_pd_port;
    return;
  }
}

void DisaggPDScheduler::step(const absl::Duration& timeout) {
  ContinuousScheduler::step(timeout);
  // send first generation token to decode instance
  // and remove the request from running_requests_ to remote_requests_map_
  if (options_.instance_role() == InstanceRole::PREFILL) {
    prefill_send_first_generation();
  }
}

bool DisaggPDScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  if (request->offline()) {
    // offline request, push to offline queue
    prefill_request_queue_offline_.enqueue(request);
    return true;
  }
  // push and wait
  prefill_request_queue_.enqueue(request);

  return true;
}

// prefill send new request to remote instance
void DisaggPDScheduler::dispatch_requests() {
  while (true) {
    const auto timeout = std::chrono::milliseconds(100);
    // Wait for online request until timeout.
    // If timeout, try to get offline request once. If no offline request,
    // continue to wait for online request. This can avoid offline request
    // blocking online request for too long time.
    std::shared_ptr<Request> request;
    if (!prefill_request_queue_.wait_dequeue_timed(request, timeout)) {
      if (!prefill_request_queue_offline_.try_dequeue(request)) {
        continue;
      }
    }

    if (request == nullptr) {
      // nullptr is a signal to exit
      break;
    }

    std::vector<std::shared_ptr<Request>> requests;
    requests.emplace_back(request);
    std::string selected_instance = "";
    proto::DisaggPDService_Stub* stub = nullptr;
    if (!request->state().decode_address.empty() && requests.size() == 1) {
      selected_instance = request->state().decode_address;
      stub = create_rpc_channel(request->state().decode_address);
    }

    // NOTE: TODO: maybe we need to support batch disatch
    // later, this meybe decrease the communication cost.
    // currently we only support one request per dispatch.

    // TODO: try to get a batch request.

    if (selected_instance.empty() && !stub) {
      // get allocated decode instance list from Master
      while (decode_inst_names_.empty()) {
        decode_inst_names_ = xservice_client_->get_static_decode_list();
        if (!decode_inst_names_.empty()) {
          LOG(INFO) << "Get PD decode instance list: "
                    << absl::StrJoin(decode_inst_names_, "; ");
          break;
        }
        sleep(1);
      }
      // select a D instance use RR currently.
      // TODO: use better decode selection strategy later. maybe different
      // strategy for offline and online request. or implement in xllm service.
      int try_decode_count = 0;
      while (!stub) {
        if (try_decode_count == decode_inst_names_.size()) {
          LOG(FATAL) << "Can not connect to all decode instances.";
        }
        ++try_decode_count;
        selected_instance = decode_inst_names_[current_decode_idx_];
        current_decode_idx_ =
            (++current_decode_idx_) % decode_inst_names_.size();
        stub = create_rpc_channel(selected_instance);
      }
    }

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      for (auto& req : requests) {
        req_to_channel_map_[req->request_id()] = stub;
      }
    }

    // TODO: send the request to the selected D instance
    // Send 'DisaggRequests' and recv 'DisaggResponses'
    xllm::proto::DisaggRequests reqs;
    xllm::proto::DisaggResponses resps;
    // prefill name (ID)
    reqs.set_prefill_name(xservice_client_->get_instance_name());
    reqs.mutable_reqs()->Reserve(requests.size());
    // currently we only support one request once.
    for (size_t i = 0; i < requests.size(); ++i) {
      // proto::DisaggRequest req;
      auto req = reqs.mutable_reqs()->Add();
      req->set_req_id(requests[i]->request_id());
      req->set_service_req_id(requests[i]->service_request_id());
      req->set_tokens_num(requests[i]->state().prompt_tokens.size());
      req->set_prompt(requests[i]->state().prompt);
      ADD_VECTOR_TO_PROTO(req->mutable_prompt_tokens(),
                          requests[i]->state().prompt_tokens);
      req->set_stream(requests[i]->state().stream);
      req->set_x_request_id(requests[i]->x_request_id());
      req->set_x_request_time(requests[i]->x_request_time());
      req->set_seq_capacity(requests[i]->state().seq_capacity);
      req->set_max_tokens(
          requests[i]->state().stopping_checker.get_max_generated_tokens());
      req->set_max_context_len(
          requests[i]->state().stopping_checker.get_max_context_len());
      req->set_ignore_eos(
          requests[i]->state().stopping_checker.get_ignore_eos());
      req->set_eos_token_id(
          requests[i]->state().stopping_checker.get_eos_token());
      if (requests[i]->state().stopping_checker.get_stop_tokens().size() > 0) {
        ADD_VECTOR_TO_PROTO(
            req->mutable_stop_token_ids(),
            requests[i]->state().stopping_checker.get_stop_tokens());
      }
      if (requests[i]->state().stopping_checker.get_stop_sequences().size() >
          0) {
        for (auto& stop_sequence :
             requests[i]->state().stopping_checker.get_stop_sequences()) {
          // proto::StopSequence proto_seq;
          auto proto_seq = req->mutable_stop_sequences()->Add();
          ADD_VECTOR_TO_PROTO(proto_seq->mutable_seq_tokens(), stop_sequence);
          //*req->mutable_stop_sequences()->Add() = proto_seq;
        }
      }
      req->set_n(requests[i]->state().n);
      req->set_best_of(requests[i]->state().best_of);
      req->set_frequency_penalty(
          requests[i]->state().sampling_param.frequency_penalty);
      req->set_presence_penalty(
          requests[i]->state().sampling_param.presence_penalty);
      req->set_repetition_penalty(
          requests[i]->state().sampling_param.repetition_penalty);
      req->set_temperature(requests[i]->state().sampling_param.temperature);
      req->set_top_p(requests[i]->state().sampling_param.top_p);
      req->set_top_k(requests[i]->state().sampling_param.top_k);
      req->set_logprobs(requests[i]->state().sampling_param.logprobs);
      req->set_top_logprobs(requests[i]->state().sampling_param.top_logprobs);
      req->set_is_embeddings(requests[i]->state().sampling_param.is_embeddings);
      req->set_echo(requests[i]->state().echo);
      req->set_skip_special_tokens(requests[i]->state().skip_special_tokens);
      //*reqs.mutable_reqs()->Add() = req;
    }
    std::vector<std::string> device_ips;
    std::vector<uint16_t> ports;
    engine_->get_device_info(device_ips, ports);
    reqs.mutable_cluster_infos()->mutable_cluster_ids()->Add(
        instance_info_.cluster_ids.begin(), instance_info_.cluster_ids.end());
    reqs.mutable_cluster_infos()->mutable_addrs()->Add(
        instance_info_.addrs.begin(), instance_info_.addrs.end());
    reqs.mutable_cluster_infos()->mutable_device_ips()->Add(device_ips.begin(),
                                                            device_ips.end());
    reqs.mutable_cluster_infos()->mutable_ports()->Add(ports.begin(),
                                                       ports.end());
    reqs.mutable_cluster_infos()->set_dp_size(options_.dp_size());

    // TODO: sync rpc here currently
    brpc::Controller cntl;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);
    // TODO: error handler
    // if (rpc failed) {
    //  // push all request back to prefill_request_queue_
    //}

    // check reqs which can not dispatch to D instance,
    // and push back to prefill_request_queue_
    CHECK_EQ(requests.size(), resps.resps().size());
    for (size_t i = 0; i < requests.size(); ++i) {
      if (resps.resps()[i].status_code() != 200) {
        // push back to prefill_request_queue_
        if (requests[i]->offline()) {
          prefill_request_queue_offline_.enqueue(requests[i]);
        } else {
          prefill_request_queue_.enqueue(requests[i]);
        }

      } else {
        for (auto& sequence : requests[i]->sequences()) {
          TransferKVInfo info;
          info.request_id = requests[i]->request_id();
          for (auto& bid : resps.resps()[i].blocks_ids()) {
            info.remote_blocks_ids.emplace_back(bid);
          }
          info.dp_rank = resps.resps()[i].dp_rank();
          // TODO: remote_instances_info_ is not multi-thread safe.
          info.remote_instance_info = remote_instances_info_[selected_instance];
          sequence->kv_state().set_transfer_kv_info(std::move(info));
        }

        // push to request_queue_, and will be executed by engine.
        request_queue_.write(requests[i]);
      }
    }
  }
}

void DisaggPDScheduler::prefill_send_first_generation() {
  if (running_sequences_.size() == 0) {
    return;
  }

  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(running_requests_.size());
  {
    std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
    for (size_t i = 0; i < running_requests_.size(); ++i) {
      auto request = running_requests_[i];
      // Check if the request is a recently completed prefill request
      if (request->sequences()[0]->num_generated_tokens() == 1) {
        if (remote_requests_map_.find(request->request_id()) !=
            remote_requests_map_.end()) {
          LOG(FATAL)
              << "Two request has the same request_id, check the requests map.";
        }
        remote_requests_map_[request->request_id()] = request;
        remote_requests_output_thread_map_[request->request_id()] =
            next_thread_idx;
        next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
        requests.emplace_back(request);

        running_requests_[i] = nullptr;
      }
    }
  }

  // No prefill request needs to be transferred to decode.
  if (requests.size() == 0) {
    return;
  }

  prefill_threadpool_.schedule([this,
                                requests = std::move(requests)]() mutable {
    // send request first token to remote instance
    // TODO: here we only support one sequence for now.
    for (auto& request : requests) {
      // TODO: support batch request later
      proto::DisaggGenerations gens;
      // proto::DisaggGeneration gen;
      auto gen = gens.mutable_gens()->Add();
      gen->set_req_id(request->request_id());
      if (request->sequences()[0]->first_token().has_value()) {
        gen->set_token_id(
            request->sequences()[0]->first_token().value().token_id);
        if (request->sequences()[0]
                ->first_token()
                .value()
                .token_logprob.has_value()) {
          gen->set_logprob(request->sequences()[0]
                               ->first_token()
                               .value()
                               .token_logprob.value());
          gen->set_has_logprob(true);
        } else {
          gen->set_has_logprob(false);
        }
        ADD_VECTOR_TO_PROTO(
            gen->mutable_top_tokens(),
            request->sequences()[0]->first_token().value().token_top_tokens);
        ADD_VECTOR_TO_PROTO(
            gen->mutable_top_logprobs(),
            request->sequences()[0]->first_token().value().token_top_logprobs);
      }
      gen->set_kv_cache_transfer_mode(options_.kv_cache_transfer_mode());
      if (options_.kv_cache_transfer_mode() == "PULL") {
        ADD_VECTOR_TO_PROTO(gen->mutable_cluster_ids(),
                            instance_info_.cluster_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_addrs(), instance_info_.addrs);
        ADD_VECTOR_TO_PROTO(gen->mutable_k_cache_ids(),
                            instance_info_.k_cache_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_v_cache_ids(),
                            instance_info_.v_cache_ids);

        const auto blocks = request->sequences()[0]->kv_state().kv_blocks();
        std::vector<uint64_t> block_ids;
        block_ids.reserve(blocks.size());
        for (const auto& block : blocks) {
          block_ids.push_back(block.id());
        }
        ADD_VECTOR_TO_PROTO(gen->mutable_block_ids(), block_ids);
        gen->set_dp_size(instance_info_.dp_size);
        gen->set_dp_rank(request->sequences()[0]->dp_rank());
      }
      //*gens.mutable_gens()->Add() = gen;

      // send first gens to remote instance
      proto::DisaggPDService_Stub* stub = nullptr;
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        // now we only support one request once.
        stub = req_to_channel_map_[request->request_id()];
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      stub->FirstGeneration(&cntl, &gens, &resp, nullptr);
      if (options_.enable_decode_response_to_service() || cntl.Failed() ||
          !resp.ok()) {
        if (cntl.Failed() || !resp.ok()) {
          LOG(ERROR) << "Failed to send first generation, " << cntl.ErrorText()
                     << ", staus: " << resp.ok();
        }
        {
          std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
          remote_requests_map_.erase(request->request_id());
          remote_requests_output_thread_map_.erase(request->request_id());
        }
        {
          std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
          req_to_channel_map_.erase(request->request_id());
        }
        kv_cache_manager_client_->deallocate(request.get());
      } else {
        // release the memory for other requests.
        // TODO: FIXME
        // Here, we should decide whether to recycle the allocated blocks
        // according to whether all the blocks have been transmitted or not.
        kv_cache_manager_client_->deallocate(request.get());
      }
    }
  });
}

bool DisaggPDScheduler::prefill_recv_generation(const RequestOutput& output) {
  std::shared_ptr<Request> request = nullptr;
  int request_thread_idx = -1;
  {
    std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
    auto it = remote_requests_map_.find(output.request_id);
    if (it == remote_requests_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << output.request_id;
      return false;
    }
    request = it->second;

    auto it2 = remote_requests_output_thread_map_.find(output.request_id);
    if (it2 == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Failed to find request thread, request id: "
                 << output.request_id;
      return false;
    }
    request_thread_idx = it2->second;
  }

  output_threadpools_[request_thread_idx].schedule(
      [this, request, output = std::move(output)]() mutable {
        if (!request->state().output_func(output) || output.finished) {
          // cancel the request if on_stream returns false
          if (!output.finished) {
            request->set_cancel();
          }
          {
            std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
            remote_requests_map_.erase(output.request_id);
            remote_requests_output_thread_map_.erase(output.request_id);
          }
          {
            std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
            req_to_channel_map_.erase(output.request_id);
          }
        }
      });

  return true;
}

// request is received from prefill
bool DisaggPDScheduler::decode_schedule(
    std::shared_ptr<Request>& request,
    const std::string& prefill_instance_name) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  proto::DisaggPDService_Stub* stub = create_rpc_channel(prefill_instance_name);
  if (!stub) {
    LOG(ERROR) << "Failed to create rpc channel for prefill instance: "
               << prefill_instance_name;
    kv_cache_manager_client_->deallocate(request.get());
    return false;
  }

  // TODO: check request_id, duplicate ids are not allowed
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    if (received_request_map_.find(request->request_id()) !=
        received_request_map_.end()) {
      LOG(FATAL) << "Decode receive same request_id from prefill.";
    }
    received_request_map_[request->request_id()] = request;
    received_request_output_thread_map_[request->request_id()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    req_to_channel_map_[request->request_id()] = stub;
    // allocate response thread to prefill instance stub.
    if (remote_prefill_thread_map_.find(stub) ==
        remote_prefill_thread_map_.end()) {
      remote_prefill_thread_map_[stub] = next_prefill_thread_idx;
      next_prefill_thread_idx = (++next_prefill_thread_idx) % kOutputTheadNum_;
    }
  }

  return true;
}

bool DisaggPDScheduler::decode_recv_first_generation(
    const std::string& req_id,
    int64_t token_id,
    bool has_logprob,
    float logprob,
    std::vector<int64_t> top_tokens,
    std::vector<float> top_logprobs,
    const std::string& kv_cache_transfer_mode,
    std::vector<uint64_t> src_cluster_ids,
    std::vector<std::string> src_addrs,
    std::vector<int64_t> src_k_cache_ids,
    std::vector<int64_t> src_v_cache_ids,
    std::vector<uint64_t> src_block_ids,
    int32_t src_dp_size,
    int32_t src_dp_rank) {
  // push to request_queue_, and will be executed by engine.
  std::shared_ptr<Request> request = nullptr;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << req_id;
      return false;
    }
    request = it->second;
    received_request_map_.erase(it);
  }

  Token first_token(token_id);
  if (has_logprob) {
    first_token.logprob = logprob;
    if (!top_tokens.empty() && !top_logprobs.empty()) {
      // NOTE: slice vector here, to avoid copy
      // so we need keep the vector `top_tokens` and `top_logprobs` lifetime
      first_token.top_tokens = top_tokens;
      first_token.top_logprobs = top_logprobs;
    }
  }
  // Enable checking whether to skip the prefill token
  if (request->state().stream) {
    request->sequences()[0]->enable_checking_prefill_token();
  }

  // TODO: we only support one sequence for currently.
  if (enable_schedule_overlap()) {
    Token fake_token(-1);
    request->sequences()[0]->append_token(fake_token);
    request->sequences()[0]->update_last_step_token(first_token);
  } else {
    request->sequences()[0]->append_token(first_token);
  }

  // pull kv cache
  if (kv_cache_transfer_mode == "PULL") {
    const auto blocks = request->sequences()[0]->kv_state().kv_blocks();
    std::vector<uint64_t> dst_block_ids;
    dst_block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      dst_block_ids.push_back(block.id());
    }

    int32_t dst_dp_rank = request->sequences()[0]->dp_rank();
    engine_->pull_kv_blocks(src_dp_size,
                            src_dp_rank,
                            src_cluster_ids,
                            src_addrs,
                            src_k_cache_ids,
                            src_v_cache_ids,
                            src_block_ids,
                            dst_dp_rank,
                            dst_block_ids);
  }

  request_queue_.write(request);
  return true;
}

bool DisaggPDScheduler::decode_send_stream_generation(
    const RequestOutput& output) {
  int request_thread_idx = -1;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_output_thread_map_.find(output.request_id);
    if (it == received_request_output_thread_map_.end()) {
      LOG(ERROR) << "Failed to find request thread, request id: "
                 << output.request_id;
      return false;
    }
    request_thread_idx = it->second;
  }

  proto::DisaggPDService_Stub* stub = nullptr;
  {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    stub = req_to_channel_map_[output.request_id];
  }
  if (!stub) {
    LOG(ERROR) << "Can not connect to remote prefill server.";
    return false;
  }

  output_threadpools_[request_thread_idx].schedule([this,
                                                    stub,
                                                    output = std::move(
                                                        output)]() mutable {
    // build proto::DisaggStreamGeneration
    proto::DisaggStreamGeneration req;
    req.set_req_id(output.request_id);
    req.set_service_req_id(output.service_request_id);
    if (output.status.has_value()) {
      auto gen_status = req.mutable_gen_status();
      gen_status->set_status_code(
          static_cast<int32_t>(output.status.value().code()));
      gen_status->set_status_msg(output.status.value().message());
    }
    req.set_finished(output.finished);
    if (output.usage.has_value()) {
      proto::OutputUsage* proto_usage = req.mutable_usage();
      proto_usage->set_num_prompt_tokens(
          output.usage.value().num_prompt_tokens);
      proto_usage->set_num_generated_tokens(
          output.usage.value().num_generated_tokens);
      proto_usage->set_num_total_tokens(output.usage.value().num_total_tokens);
    }
    req.mutable_outputs()->Reserve(output.outputs.size());
    for (auto& seq_output : output.outputs) {
      // proto::SequenceOutput proto_seq_out;
      auto proto_seq_out = req.mutable_outputs()->Add();
      proto_seq_out->set_index(seq_output.index);
      proto_seq_out->set_text(seq_output.text);
      if (seq_output.finish_reason.has_value()) {
        proto_seq_out->set_finish_reason(seq_output.finish_reason.value());
      } else {
        proto_seq_out->set_finish_reason("");
      }
      ADD_VECTOR_TO_PROTO(proto_seq_out->mutable_token_ids(),
                          seq_output.token_ids);
      if (seq_output.logprobs.has_value()) {
        size_t logprobs_size = seq_output.logprobs.value().size();
        proto_seq_out->mutable_logprobs()->Reserve(logprobs_size);
        for (size_t i = 0; i < logprobs_size; ++i) {
          // proto::LogProb logprob;
          auto logprob = proto_seq_out->mutable_logprobs()->Add();
          proto::LogProbData* log_prob_data = logprob->mutable_log_prob_data();
          log_prob_data->set_token(seq_output.logprobs.value()[i].token);
          log_prob_data->set_token_id(seq_output.logprobs.value()[i].token_id);
          log_prob_data->set_logprob(seq_output.logprobs.value()[i].logprob);
          log_prob_data->set_finished_token(
              seq_output.logprobs.value()[i].finished_token);
          if (seq_output.logprobs.value()[i].top_logprobs.has_value()) {
            size_t top_logprobs_size =
                seq_output.logprobs.value()[i].top_logprobs.value().size();
            for (size_t j = 0; j < top_logprobs_size; ++j) {
              proto::LogProbData* top_log_prob_data =
                  logprob->mutable_top_logprobs()->Add();
              top_log_prob_data->set_token(
                  seq_output.logprobs.value()[i].top_logprobs.value()[j].token);
              top_log_prob_data->set_token_id(seq_output.logprobs.value()[i]
                                                  .top_logprobs.value()[j]
                                                  .token_id);
              top_log_prob_data->set_logprob(seq_output.logprobs.value()[i]
                                                 .top_logprobs.value()[j]
                                                 .logprob);
              top_log_prob_data->set_finished_token(
                  seq_output.logprobs.value()[i]
                      .top_logprobs.value()[j]
                      .finished_token);
            }
          }
          //*proto_seq_out.mutable_logprobs()->Add() = logprob;
        }
      }
      //*req.mutable_outputs()->Add() = proto_seq_out;
    }

    // Sync
    proto::Status resp;
    brpc::Controller cntl;
    stub->Generation(&cntl, &req, &resp, nullptr);
    // TODO: error handler

    // TODO: handle resp status

    if (output.finished) {
      std::lock_guard<std::mutex> lock(received_request_map_mutex_);
      received_request_output_thread_map_.erase(output.request_id);
    }
  });

  return true;
}

std::vector<bool> DisaggPDScheduler::decode_send_stream_generations(
    const std::vector<RequestOutput>& outputs) {
  std::vector<bool> send_status;
  send_status.resize(outputs.size(), true);

  // 1. response to xllm service to avoid the redirect cost.
  if (options_.enable_decode_response_to_service()) {
    output_threadpools_[0].schedule(
        [this, outputs = std::move(outputs)]() mutable {
          xservice_client_->generations(outputs);
          // TODO: error handler
          // TODO: handle resp status
          {
            std::lock_guard<std::mutex> lock(received_request_map_mutex_);
            for (auto& output : outputs) {
              if (output.finished) {
                received_request_output_thread_map_.erase(output.request_id);
              }
            }
          }
        });
    return send_status;
  }

  // 2. response to prefill instance
  // find all prefill stubs
  // record the indexes for each prefill stub
  std::unordered_map<proto::DisaggPDService_Stub*, std::vector<RequestOutput>>
      per_outputs;
  std::unordered_map<proto::DisaggPDService_Stub*, std::vector<int>>
      per_outputs_idx;
  {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto it = req_to_channel_map_.find(outputs[i].request_id);
      if (it == req_to_channel_map_.end()) {
        LOG(ERROR) << "Can not connect to remote prefill server, request is "
                   << outputs[i].request_id;
        send_status[i] = false;
        continue;
      }
      proto::DisaggPDService_Stub* req_stub = it->second;
      if (per_outputs.find(req_stub) == per_outputs.end()) {
        std::vector<RequestOutput> o;
        o.emplace_back(outputs[i]);
        per_outputs[req_stub] = std::move(o);
        per_outputs_idx[req_stub] = {i};
      } else {
        per_outputs[req_stub].emplace_back(std::move(outputs[i]));
        per_outputs_idx[req_stub].emplace_back(i);
      }
    }
  }

  // create proto and send outputs to prefill
  for (auto& o : per_outputs) {
    int request_thread_idx = -1;
    {
      std::lock_guard<std::mutex> lock(received_request_map_mutex_);
      auto it = remote_prefill_thread_map_.find(o.first);
      if (it == remote_prefill_thread_map_.end()) {
        LOG(ERROR) << "Failed to find prefill stub thread, stub: " << o.first;
        for (auto idx : per_outputs_idx[o.first]) {
          send_status[idx] = false;
        }
        continue;
      }

      request_thread_idx = it->second;
    }

    output_threadpools_[request_thread_idx].schedule([this,
                                                      stub = o.first,
                                                      outputs = std::move(
                                                          o.second)]() mutable {
      proto::DisaggStreamGenerations gens;
      for (auto& output : outputs) {
        // build proto::DisaggStreamGeneration
        proto::DisaggStreamGeneration* req = gens.mutable_gens()->Add();
        req->set_req_id(output.request_id);
        req->set_service_req_id(output.service_request_id);
        if (output.status.has_value()) {
          auto gen_status = req->mutable_gen_status();
          gen_status->set_status_code(
              static_cast<int32_t>(output.status.value().code()));
          gen_status->set_status_msg(output.status.value().message());
        }
        req->set_finished(output.finished);
        if (output.usage.has_value()) {
          proto::OutputUsage* proto_usage = req->mutable_usage();
          proto_usage->set_num_prompt_tokens(
              output.usage.value().num_prompt_tokens);
          proto_usage->set_num_generated_tokens(
              output.usage.value().num_generated_tokens);
          proto_usage->set_num_total_tokens(
              output.usage.value().num_total_tokens);
        }
        req->mutable_outputs()->Reserve(output.outputs.size());
        for (auto& seq_output : output.outputs) {
          // proto::SequenceOutput proto_seq_out;
          auto proto_seq_out = req->mutable_outputs()->Add();
          proto_seq_out->set_index(seq_output.index);
          proto_seq_out->set_text(seq_output.text);
          if (seq_output.finish_reason.has_value()) {
            proto_seq_out->set_finish_reason(seq_output.finish_reason.value());
          } else {
            proto_seq_out->set_finish_reason("");
          }
          ADD_VECTOR_TO_PROTO(proto_seq_out->mutable_token_ids(),
                              seq_output.token_ids);
          if (seq_output.logprobs.has_value()) {
            size_t logprobs_size = seq_output.logprobs.value().size();
            proto_seq_out->mutable_logprobs()->Reserve(logprobs_size);
            for (size_t i = 0; i < logprobs_size; ++i) {
              // proto::LogProb logprob;
              auto logprob = proto_seq_out->mutable_logprobs()->Add();
              proto::LogProbData* log_prob_data =
                  logprob->mutable_log_prob_data();
              log_prob_data->set_token(seq_output.logprobs.value()[i].token);
              log_prob_data->set_token_id(
                  seq_output.logprobs.value()[i].token_id);
              log_prob_data->set_logprob(
                  seq_output.logprobs.value()[i].logprob);
              log_prob_data->set_finished_token(
                  seq_output.logprobs.value()[i].finished_token);
              if (seq_output.logprobs.value()[i].top_logprobs.has_value()) {
                size_t top_logprobs_size =
                    seq_output.logprobs.value()[i].top_logprobs.value().size();
                for (size_t j = 0; j < top_logprobs_size; ++j) {
                  proto::LogProbData* top_log_prob_data =
                      logprob->mutable_top_logprobs()->Add();
                  top_log_prob_data->set_token(seq_output.logprobs.value()[i]
                                                   .top_logprobs.value()[j]
                                                   .token);
                  top_log_prob_data->set_token_id(seq_output.logprobs.value()[i]
                                                      .top_logprobs.value()[j]
                                                      .token_id);
                  top_log_prob_data->set_logprob(seq_output.logprobs.value()[i]
                                                     .top_logprobs.value()[j]
                                                     .logprob);
                  top_log_prob_data->set_finished_token(
                      seq_output.logprobs.value()[i]
                          .top_logprobs.value()[j]
                          .finished_token);
                }
              }
              //*proto_seq_out.mutable_logprobs()->Add() = logprob;
            }
          }
          //*req.mutable_outputs()->Add() = proto_seq_out;
        }
      }

      // Sync
      proto::StatusSet resp;
      brpc::Controller cntl;
      stub->Generations(&cntl, &gens, &resp, nullptr);
      // TODO: error handler

      // TODO: handle resp status

      {
        std::lock_guard<std::mutex> lock(received_request_map_mutex_);
        for (auto& output : outputs) {
          if (output.finished) {
            received_request_output_thread_map_.erase(output.request_id);
          }
        }
      }
    });
  }

  return send_status;
}

std::vector<Block> DisaggPDScheduler::allocate_raw_blocks(int token_num,
                                                          int32_t& dp_rank) {
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  if (kv_cache_manager_client_->kv_cache_utilization() <
      FLAGS_prefill_scheduling_memory_usage_threshold) {
    return allocate_blocks_for(token_num, dp_rank);
  } else {
    return {};
  }
}

void DisaggPDScheduler::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  const auto now = absl::Now();
  for (Sequence* sequence : sequences) {
    int64_t tbt_milliseconds = sequence->tbt(now);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
      recent_ttft_.emplace_back(tbt_milliseconds);
    } else {
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
      recent_tbt_.emplace_back(tbt_milliseconds);
    }
  }
}

void DisaggPDScheduler::get_latency_metrics(std::vector<int64_t>& ttft,
                                            std::vector<int64_t>& tbt) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
  ttft = std::move(recent_ttft_);
  tbt = std::move(recent_tbt_);
}

}  // namespace xllm
