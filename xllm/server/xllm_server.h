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
