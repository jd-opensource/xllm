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

#pragma once

#include <brpc/channel.h>

#include <shared_mutex>
#include <string>
#include <thread>

#include "common/etcd_client.h"
#include "forward_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request_output.h"
#include "scheduler/scheduler.h"
#include "xservice.pb.h"
namespace xllm {

struct ServiceConfig {
  ServiceConfig(bool decode_to_service)
      : enable_decode_response_to_service(decode_to_service) {}

  bool enable_decode_response_to_service = false;
};

class XServiceClient {
 public:
  static XServiceClient* get_instance() {
    static XServiceClient xservice_client;
    return &xservice_client;
  }

  ~XServiceClient();
  bool init(const std::string& etcd_addr,
            const std::string& xservice_addr,
            const std::string& instance_name = "",
            const BlockManagerPool* block_manager_pool = nullptr);
  void set_scheduler(Scheduler* scheduler);
  bool initialize_done() { return initialize_done_; }

  std::string get_instance_name();
  void register_instance(const InstanceInfo& instance_info);
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();
  std::vector<std::string> get_static_prefill_list();
  ServiceConfig get_config();

  // response generation tokens to xllm service
  void generations(const std::vector<RequestOutput>& outputs);

 private:
  void handle_master_service_watch(const etcd::Response& response);

 private:
  XServiceClient() = default;

  bool exited_ = false;
  bool register_done_ = false;
  bool initialize_done_ = false;
  std::string instance_name_;

  std::string xservice_addr_;
  std::unique_ptr<brpc::Channel> xservice_channel_;
  std::unique_ptr<xllm_service::proto::XllmRpcService_Stub> xservice_stub_;
  std::unique_ptr<std::thread> heartbeat_thread_;

  std::shared_mutex mutex_;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  const BlockManagerPool* block_manager_pool_;  // not own
  Scheduler* scheduler_;                        // not own
};

}  // namespace xllm
