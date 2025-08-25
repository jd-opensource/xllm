#pragma once

#include "disagg_pd.pb.h"

namespace xllm {

class Engine;
class Request;
class DisaggPDScheduler;

// a class to handle disagg_pd requests
class DisaggPDServiceImplInterface {
 public:
  DisaggPDServiceImplInterface() = default;
  virtual ~DisaggPDServiceImplInterface() = default;

  virtual void decode_recv_new_requests(const proto::DisaggRequests* request,
                                        proto::DisaggResponses* response) {}

  virtual void decode_recv_first_generation(
      const proto::DisaggGenerations* request,
      proto::Status* response) {}

  virtual bool prefill_recv_generation(
      const proto::DisaggStreamGeneration* request,
      proto::Status* response) {
    return true;
  }

  virtual void prefill_recv_generations(
      const proto::DisaggStreamGenerations* requests,
      proto::StatusSet* responses) {}
};

class DisaggPDServiceImpl final : public DisaggPDServiceImplInterface {
 public:
  explicit DisaggPDServiceImpl(DisaggPDScheduler* scheduler, Engine* engine);
  ~DisaggPDServiceImpl() = default;

  bool prefill_recv_generation(const proto::DisaggStreamGeneration* request,
                               proto::Status* response) override;

  void prefill_recv_generations(const proto::DisaggStreamGenerations* requests,
                                proto::StatusSet* responses) override;

  void decode_recv_new_requests(const proto::DisaggRequests* request,
                                proto::DisaggResponses* response) override;

  void decode_recv_first_generation(const proto::DisaggGenerations* request,
                                    proto::Status* response) override;

 private:
  std::shared_ptr<Request> generate_request(const proto::DisaggRequest& req);

  DisaggPDScheduler* scheduler_;  // not owned
  Engine* engine_;                // not owned
};

}  // namespace xllm
