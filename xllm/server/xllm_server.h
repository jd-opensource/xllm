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

#include "api_service/api_service.h"
#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/disagg_pd_service.h"
#include "core/distributed_runtime/worker_service.h"

namespace xllm {

class XllmServer final {
 public:
  XllmServer();
  ~XllmServer();

  bool start(std::unique_ptr<APIService> api_service);
  bool start(std::unique_ptr<DisaggPDService> disagg_pd_service);
  bool start(std::shared_ptr<CollectiveService> service,
             const std::string& addr);
  bool start(std::shared_ptr<WorkerService> service, const std::string& addr);

  void run();
  void stop();

  bool has_initialized() const { return has_initialized_; }
  std::string listen_address() const { return listen_address_; }
  int listen_port() const { return listen_port_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(XllmServer);
  bool create_server(google::protobuf::Service* service,
                     const std::string& addr,
                     int port,
                     const std::string& server_name);

 private:
  bool has_initialized_ = false;
  int listen_port_ = -1;
  std::string listen_address_;
  std::unique_ptr<brpc::Server> server_;
  std::unique_ptr<std::thread> running_thread_;
};

class XllmServerFactory {
 public:
  static std::unique_ptr<XllmServer> create_xllm_server() {
    return std::make_unique<XllmServer>();
  }

 private:
  XllmServerFactory() = default;
  ~XllmServerFactory() = default;
  DISALLOW_COPY_AND_ASSIGN(XllmServerFactory);
};

}  // namespace xllm
