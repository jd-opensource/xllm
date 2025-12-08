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

#include "xtensor_manager_server.h"

#include <brpc/channel.h>

#include "common/global_flags.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"
#include "xtensor_manager.h"
#include "xtensor_manager_service.h"

namespace xllm {
void XTensorManagerServer::create_server(const xtensor::Options& options,
                                         std::atomic<bool>& done,
                                         const std::string& master_node_addr,
                                         const torch::Device& d,
                                         int world_size,
                                         int global_rank,
                                         int32_t dp_size,
                                         int local_rank) {
  Device device(d);
  device.set_device();

  auto xtensor_manager_global_rank = global_rank;
  auto xtensor_manager_service = std::make_shared<XTensorManagerService>(
      xtensor_manager_global_rank, world_size, d);

  auto addr = net::get_local_ip_addr();
  auto xtensor_manager_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!xtensor_manager_server->start(xtensor_manager_service, addr + ":0")) {
    LOG(ERROR)
        << "failed to start distribute xtensor manager server on address: "
        << addr;
    return;
  }

  auto xtensor_manager_server_addr =
      addr + ":" + std::to_string(xtensor_manager_server->listen_port());
  LOG(INFO) << "XTensorManager " << xtensor_manager_global_rank
            << ": server address: " << xtensor_manager_server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(xtensor_manager_server_addr);
  addr_info.set_global_rank(xtensor_manager_global_rank);
  proto::CommUniqueIdList uids;
  sync_master_node(master_node_addr, addr_info, uids);

  std::unique_ptr<XTensorManager> xtensor_manager =
      std::make_unique<XTensorManager>(options, d);
  xtensor_manager_service->set_xtensor_manager(std::move(xtensor_manager));

  done.store(true);

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  xtensor_manager_server->run();
}

XTensorManagerServer::XTensorManagerServer(int local_xtensor_manager_idx,
                                           const std::string& master_node_addr,
                                           std::atomic<bool>& done,
                                           const torch::Device& device,
                                           const xtensor::Options& options)
    : server_name_("DistributeXTensorManagerServer") {
  server_name_.append(std::to_string(options.server_idx()));
  xtensor_manager_thread_ =
      std::make_unique<std::thread>([this,
                                     &options,
                                     &done,
                                     &master_node_addr,
                                     &device,
                                     local_xtensor_manager_idx] {
        create_server(options,
                      done,
                      master_node_addr,
                      device,
                      /*num_shards=*/0,
                      /*block_size=*/0,
                      /*max_blocks=*/0,
                      /*port=*/local_xtensor_manager_idx);
      });
}

bool XTensorManagerServer::sync_master_node(const std::string& master_node_addr,
                                            proto::AddressInfo& addr_info,
                                            proto::CommUniqueIdList& uids) {
  // Brpc connection resources
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = 10000;
  options.max_retry = 3;
  if (channel.Init(master_node_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Failed to initialize BRPC channel to " << master_node_addr;
    return false;
  }
  proto::Collective_Stub stub(&channel);

  // Retry until master node ready
  int try_count = 0;
  brpc::Controller cntl;
  const int sleep_time_second = 3;
  while (try_count < FLAGS_max_reconnect_count) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, NULL);
    if (cntl.Failed()) {
      LOG(WARNING) << "XTensorManager#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    } else {
      LOG(INFO) << "XTensorManager#" << addr_info.global_rank()
                << " connect to " << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >= FLAGS_max_reconnect_count) {
    LOG(ERROR) << "XTensorManager#" << addr_info.global_rank() << " connect to "
               << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

XTensorManagerServer::~XTensorManagerServer() {
  if (xtensor_manager_thread_->joinable()) {
    xtensor_manager_thread_->join();
  }
}
}  // namespace xllm