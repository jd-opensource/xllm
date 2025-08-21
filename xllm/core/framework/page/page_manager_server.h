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

#include <brpc/server.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <thread>

#include "common/macros.h"
#include "options.h"
#include "page_manager.pb.h"

namespace xllm {
class PageManagerServer {
 public:
  PageManagerServer(int local_page_manager_idx,
                    const std::string& master_node_addr,
                    std::atomic<bool>& done,
                    const torch::Device& device,
                    const page::Options& options);
  virtual ~PageManagerServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerServer);

  void create_server(const page::Options& options,
                     std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& device,
                     int world_size,
                     int global_rank,
                     int32_t dp_size,
                     int local_rank);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> page_manager_thread_;
};
}  // namespace xllm