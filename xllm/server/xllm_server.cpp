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

#include "xllm_server.h"

#include <brpc/server.h>
#include <butil/at_exit.h>

#include "core/common/global_flags.h"

namespace xllm {

XllmServer::XllmServer() { butil::AtExitManager exit_manager; }

XllmServer::~XllmServer() {
  if (running_thread_ && running_thread_->joinable()) {
    running_thread_->join();
  }
  stop();
}

bool XllmServer::start(std::unique_ptr<APIService> service) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service.get(),
                          brpc::SERVER_DOESNT_OWN_SERVICE,
                          "v1/completions => CompletionsHttp,"
                          "v1/chat/completions => ChatCompletionsHttp,"
                          "v1/embeddings => EmbeddingsHttp,"
                          "v1/models => ModelsHttp,"
                          "get_cache_info => GetCacheInfo,"
                          "link_cluster => LinkCluster,"
                          "unlink_cluster => UnlinkCluster,"
                          "v2/repository/index => ModelVersionsHttp,") != 0) {
    LOG(ERROR) << "Fail to add api service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  if (server_->Start(FLAGS_port, &options) != 0) {
    LOG(ERROR) << "Failed to start server on port " << FLAGS_port;
    return false;
  }
  LOG(INFO) << "Brpc Server started on port " << FLAGS_port
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  listen_address_ =
      std::string(butil::endpoint2str(server_->listen_address()).c_str());
  listen_port_ = FLAGS_port;
  has_initialized_ = true;
  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server_->RunUntilAskedToQuit();

  return true;
}

bool XllmServer::start(std::unique_ptr<DisaggPDService> service) {
  std::string addr("");
  if (!FLAGS_host.empty()) {
    addr = FLAGS_host + ":" + std::to_string(FLAGS_disagg_pd_port);
  }
  if (!create_server((google::protobuf::Service*)(service.get()),
                     addr,
                     FLAGS_disagg_pd_port,
                     "Disagg PD")) {
    return false;
  }

  has_initialized_ = true;
  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server_->RunUntilAskedToQuit();
  return true;
}

bool XllmServer::start(std::shared_ptr<CollectiveService> service,
                       const std::string& addr) {
  if (!create_server((google::protobuf::Service*)(service.get()),
                     addr,
                     -1,
                     "Collective")) {
    return false;
  }

  running_thread_ =
      std::make_unique<std::thread>([this, service = std::move(service)]() {
        has_initialized_ = true;
        server_->RunUntilAskedToQuit();
      });

  return true;
}

bool XllmServer::start(std::shared_ptr<WorkerService> service,
                       const std::string& addr) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Fail to add DistributeWorker service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  listen_address_ = addr;
  if (server_->Start(addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to start distribute server on address: " << addr;
    return false;
  }
  listen_port_ = server_->listen_address().port;
  LOG(INFO) << "DistributeWorker started on address "
            << server_->listen_address()
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  return true;
}

bool XllmServer::start(std::shared_ptr<PageManagerService> service,
                       const std::string& addr) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Fail to add DistributePageManager service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  listen_address_ = addr;
  if (server_->Start(addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to start distribute server on address: " << addr;
    return false;
  }
  listen_port_ = server_->listen_address().port;
  LOG(INFO) << "DistributePageManager started on address "
            << server_->listen_address()
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  return true;
}

bool XllmServer::create_server(google::protobuf::Service* service,
                               const std::string& addr,
                               int port,
                               const std::string& server_name) {
  server_ = std::make_unique<brpc::Server>();
  if (server_->AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Fail to add " << server_name << " service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  butil::EndPoint endpoint;
  if (!addr.empty()) {
    listen_address_ = addr;
    if (butil::str2endpoint(listen_address_.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert listen_address_ to endpoint failed: "
                 << listen_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, port);
    listen_address_ =
        std::string(butil::endpoint2str(server_->listen_address()).c_str());
  }
  listen_port_ = port > 0 ? port : server_->listen_address().port;

  if (server_->Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Failed to start " << server_name
               << " server on address: " << endpoint;
    return false;
  }
  LOG(INFO) << server_name << " server started on address " << endpoint
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  return true;
}

void XllmServer::run() {
  if (has_initialized_) {
    return;
  }

  has_initialized_ = true;
  server_->RunUntilAskedToQuit();
}

void XllmServer::stop() { return; }

}  // namespace xllm
