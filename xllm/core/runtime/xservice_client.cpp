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

#include "xservice_client.h"

#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include "util/hash_util.h"
#include "util/net.h"

namespace xllm {
namespace {
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";
static std::unordered_map<xllm_service::proto::InstanceType, std::string>
    ETCD_KEYS_PREFIX_MAP = {
        {xllm_service::proto::InstanceType::DEFAULT, "XLLM:DEFAULT:"},
        {xllm_service::proto::InstanceType::PREFILL, "XLLM:PREFILL:"},
        {xllm_service::proto::InstanceType::DECODE, "XLLM:DECODE:"},
        {xllm_service::proto::InstanceType::MIX, "XLLM:MIX:"},
};

std::string parse_instance_name(const std::string& name) {
  if (name.empty()) return "";
  // Vlidate the format of instance name
  // The format is `ip:port` currently.
  auto pos = name.find(':');
  if (pos == std::string::npos) {
    // only offer the port, we need to fill the ip address
    return xllm::net::get_local_ip_addr() + ":" + name;
  }
  return name;
}

bool check_instance_name(const std::string& name) {
  std::vector<std::string> addr = absl::StrSplit(name, ':');
  // Now only support `ip:port` format
  if (addr.size() != 2 || addr[0].empty() || addr[1].empty()) {
    LOG(ERROR)
        << "Invalid instance name format, now only support `ip:port` style.";
    return false;
  }

  return true;
}

}  // namespace

bool XServiceClient::init(const std::string& etcd_addr,
                          const std::string& instance_name,
                          const BlockManagerPool* block_manager_pool) {
  if (etcd_addr.empty()) {
    LOG(ERROR) << "etcd_addr address is empty.";
    return false;
  }

  instance_name_ = instance_name;
  chan_options_.protocol = "http";
  chan_options_.max_retry = 3;
  chan_options_.timeout_ms = FLAGS_rpc_channel_timeout_ms;

  etcd_client_ = std::make_unique<EtcdClient>(etcd_addr);

  while (!etcd_client_->get_master_service(ETCD_MASTER_SERVICE_KEY,
                                           &xservice_addr_)) {
    LOG(ERROR) << "Master service not set, wait 2s!";
    sleep(2);
  }

  if (!check_instance_name(xservice_addr_)) {
    LOG(FATAL) << "Invalid master service name format, now only support "
                  "`ip:port` style.";
    return false;
  }

  xservice_channel_ = std::make_unique<brpc::Channel>();
  if (xservice_channel_->Init(xservice_addr_.c_str(), "", &chan_options_) !=
      0) {
    LOG(FATAL) << "Fail to initialize xsevrice channel to server "
               << xservice_addr_;
    return false;
  }
  xservice_stub_ = std::make_unique<xllm_service::proto::XllmRpcService_Stub>(
      xservice_channel_.get());

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&XServiceClient::heartbeat, this);

  auto func = std::bind(&XServiceClient::handle_master_service_watch,
                        this,
                        std::placeholders::_1);

  etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, func);

  block_manager_pool_ = block_manager_pool;

  initialize_done_ = true;
  return true;
}

void XServiceClient::set_scheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
}

XServiceClient::~XServiceClient() {
  exited_ = true;
  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
}

std::string XServiceClient::get_instance_name() { return instance_name_; }

void XServiceClient::register_instance(const InstanceInfo& instance_info) {
  std::string key_prefix = "";
  if (InstanceRole(instance_info.type) == InstanceRole::DEFAULT) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DEFAULT];
  } else if (InstanceRole(instance_info.type) == InstanceRole::PREFILL) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::PREFILL];
  } else if (InstanceRole(instance_info.type) == InstanceRole::DECODE) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DECODE];
  } else if (InstanceRole(instance_info.type) == InstanceRole::MIX) {
    key_prefix = ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::MIX];
  } else {
    LOG(ERROR) << "Unsupported instance type: " << instance_info.type;
    return;
  }

  int retry_cnt = 0;
  while (
      !etcd_client_->register_instance(key_prefix.append(instance_info.name),
                                       instance_info.serialize_to_json().dump(),
                                       FLAGS_etcd_ttl)) {
    if (retry_cnt >= 30) {
      LOG(FATAL) << "Register Instance to etcd faill!";
      return;
    }

    LOG(ERROR) << "Register Instance faill, wait 2s!";
    sleep(2);
    retry_cnt++;
  }

  register_done_ = true;
  LOG(INFO) << "Success register instance to etcd.";
}

InstanceInfo XServiceClient::get_instance_info(
    const std::string& instance_name) {
  InstanceInfo result;
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceMetaInfo resp;
  req.set_name(instance_name);
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    xservice_stub_->GetInstanceInfo(&cntl, &req, &resp, nullptr);
  }
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get instance info from xservice server "
               << xservice_addr_ << ", error text: " << cntl.ErrorText();
    return result;
  }
  result.name = resp.name();
  result.rpc_address = resp.rpc_address();
  if (resp.type() == xllm_service::proto::InstanceType::PREFILL) {
    result.type = "PREFILL";
  } else if (resp.type() == xllm_service::proto::InstanceType::DECODE) {
    result.type = "DECODE";
  } else if (resp.type() == xllm_service::proto::InstanceType::MIX) {
    result.type = "MIX";
  } else {
    result.type = "DEFAULT";
  }
  // parse kv cache info
  for (auto& cluster_id : resp.cluster_ids()) {
    result.cluster_ids.emplace_back(cluster_id);
  }
  for (auto& addr : resp.addrs()) {
    result.addrs.emplace_back(addr);
  }
  for (auto& k_cache_id : resp.k_cache_ids()) {
    result.k_cache_ids.emplace_back(k_cache_id);
  }
  for (auto& v_cache_id : resp.v_cache_ids()) {
    result.v_cache_ids.emplace_back(v_cache_id);
  }
  result.dp_size = resp.dp_size();
  for (auto& ip : resp.device_ips()) {
    result.device_ips.emplace_back(ip);
  }
  for (auto& port : resp.ports()) {
    result.ports.emplace_back(port);
  }

  return result;
}

void XServiceClient::heartbeat() {
  KvCacheEvent event;
  while (!exited_) {
    event.clear();
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(FLAGS_heart_beat_interval * 1000)));
    if (!register_done_) continue;
    if (block_manager_pool_ == nullptr || scheduler_ == nullptr) continue;

    brpc::Controller cntl;
    xllm_service::proto::HeartbeatRequest req;
    req.set_name(instance_name_);
    if (block_manager_pool_->options().enable_prefix_cache()) {
      block_manager_pool_->get_merged_kvcache_event(&event);
      auto cache_event = req.mutable_cache_event();
      if (event.stored_cache.size()) {
        cache_event->mutable_stored_cache()->Reserve(event.stored_cache.size());
        for (auto& hash_key : event.stored_cache) {
          cache_event->add_stored_cache(hash_key.data, sizeof(hash_key.data));
        }
      }

      if (event.removed_cache.size()) {
        cache_event->mutable_removed_cache()->Reserve(
            event.removed_cache.size());
        for (auto& hash_key : event.removed_cache) {
          cache_event->add_removed_cache(hash_key.data, sizeof(hash_key.data));
        }
      }
    }

    req.mutable_load_metrics()->set_gpu_cache_usage_perc(
        block_manager_pool_->get_gpu_cache_usage_perc());

    req.mutable_load_metrics()->set_waiting_requests_num(
        scheduler_->get_waiting_requests_num());

    std::vector<int64_t> ttft;
    std::vector<int64_t> tbt;
    scheduler_->get_latency_metrics(ttft, tbt);
    if (!ttft.empty()) {
      auto max_ttft = std::max_element(ttft.begin(), ttft.end());
      req.mutable_latency_metrics()->set_recent_max_ttft(*max_ttft);
    }

    if (!tbt.empty()) {
      auto max_tbt = std::max_element(tbt.begin(), tbt.end());
      req.mutable_latency_metrics()->set_recent_max_tbt(*max_tbt);
    }

    xllm_service::proto::Status resp;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      xservice_stub_->Heartbeat(&cntl, &req, &resp, nullptr);
    }
    if (cntl.Failed()) {
      LOG(ERROR) << "Failed to send heartbeat to xservice " << xservice_addr_
                 << ", error msg is: " << cntl.ErrorText();
    } else if (!resp.ok()) {
      LOG(ERROR) << "Failed to send heartbeat to xservice " << xservice_addr_;
    }
  }
}

std::vector<std::string> XServiceClient::get_static_decode_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    xservice_stub_->GetStaticDecodeList(&cntl, &req, &resp, nullptr);
  }
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static decode list from xservice server "
               << xservice_addr_ << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_static_prefill_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    xservice_stub_->GetStaticPrefillList(&cntl, &req, &resp, nullptr);
  }
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static prefill list from xservice server "
               << xservice_addr_ << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

void XServiceClient::generations(const std::vector<RequestOutput>& outputs) {
  // response to xllm service to avoid the redirect cost.
  proto::DisaggStreamGenerations gens;
  for (auto& output : outputs) {
    // build xllm_service::proto::DisaggStreamGeneration
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
      proto_usage->set_num_total_tokens(output.usage.value().num_total_tokens);
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
      proto_seq_out->mutable_token_ids()->Reserve(seq_output.token_ids.size());
      for (const auto& value : seq_output.token_ids) {
        *proto_seq_out->mutable_token_ids()->Add() = value;
      }
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
  }
  proto::StatusSet resp;
  brpc::Controller cntl;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    xservice_stub_->Generations(&cntl, &gens, &resp, nullptr);
  }
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to response tokens to xservice server "
               << xservice_addr_ << ", error text: " << cntl.ErrorText();
  }
  // handle error here
}

void XServiceClient::handle_master_service_watch(
    const etcd::Response& response) {
  if (response.events().empty() || exited_) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() == etcd::Event::EventType::PUT) {
      auto xservice_addr = event.kv().as_string();

      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (xservice_addr_.compare(xservice_addr) != 0) {
        xservice_addr_ = xservice_addr;
        xservice_channel_ = std::make_unique<brpc::Channel>();
        if (xservice_channel_->Init(
                xservice_addr_.c_str(), "", &chan_options_) != 0) {
          LOG(FATAL) << "Fail to initialize xsevrice channel to server "
                     << xservice_addr_;
          return;
        }
        xservice_stub_ =
            std::make_unique<xllm_service::proto::XllmRpcService_Stub>(
                xservice_channel_.get());
        LOG(INFO) << "Change MasterService to " << xservice_addr_;
      }
    }
  }
}

}  // namespace xllm
