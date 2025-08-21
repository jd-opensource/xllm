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

#include "page_manager_server.h"

#include <brpc/channel.h>

#include "common/global_flags.h"
#include "page_manager.h"
#include "page_manager_service.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"

namespace xllm {
void PageManagerServer::create_server(const page::Options& options,
                                      std::atomic<bool>& done,
                                      const std::string& master_node_addr,
                                      const torch::Device& device,
                                      int world_size,
                                      int global_rank,
                                      int32_t dp_size,
                                      int local_rank) {
  int device_id = device.index();
#if defined(USE_NPU)
  int ret = aclrtSetDevice(device_id);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id: " << device_id << " failed, ret:" << ret;
  }
#endif

  auto page_manager_global_rank = global_rank;
  auto page_manager_service = std::make_shared<PageManagerService>(
      page_manager_global_rank, world_size, device);

  auto addr = net::get_local_ip_addr();
  auto page_manager_server = ServerRegistry::get_instance().register_server(
      "DistributePageManagerServer");
  if (!page_manager_server->start(page_manager_service, addr + ":0")) {
    LOG(ERROR) << "failed to start distribute page manager server on address: "
               << addr;
    return;
  }

  auto page_manager_server_addr =
      addr + ":" + std::to_string(page_manager_server->listen_port());
  LOG(INFO) << "PageManager " << page_manager_global_rank
            << ": server address: " << page_manager_server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(page_manager_server_addr);
  addr_info.set_global_rank(page_manager_global_rank);
  proto::CommUniqueIdList uids;
  sync_master_node(master_node_addr, addr_info, uids);

  std::unique_ptr<PageManager> page_manager =
      std::make_unique<PageManager>(options, device);
  page_manager_service->set_page_manager(std::move(page_manager));

  done.store(true);

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  page_manager_server->run();
}

PageManagerServer::PageManagerServer(int local_page_manager_idx,
                                     const std::string& master_node_addr,
                                     std::atomic<bool>& done,
                                     const torch::Device& device,
                                     const page::Options& options) {
  page_manager_thread_ =
      std::make_unique<std::thread>([this,
                                     &options,
                                     &done,
                                     &master_node_addr,
                                     &device,
                                     local_page_manager_idx] {
        create_server(options,
                      done,
                      master_node_addr,
                      device,
                      /*num_shards=*/0,
                      /*block_size=*/0,
                      /*max_blocks=*/0,
                      /*port=*/local_page_manager_idx);
      });
}

bool PageManagerServer::sync_master_node(const std::string& master_node_addr,
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
  while (try_count < FLAGS_max_connect_count) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, NULL);
    if (cntl.Failed()) {
      LOG(WARNING) << "PageManager#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_sleep_time_second));
    } else {
      LOG(INFO) << "PageManager#" << addr_info.global_rank() << " connect to "
                << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "PageManager#" << addr_info.global_rank() << " connect to "
               << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

PageManagerServer::~PageManagerServer() {
  if (page_manager_thread_->joinable()) {
    page_manager_thread_->join();
  }
}
}  // namespace xllm