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

#include "disagg_pd.pb.h"

namespace xllm {

class Engine;
class Request;
class PDOOCScheduler;

// a class to handle disagg_pd requests
class PDOOCServiceImplInterface {
 public:
  PDOOCServiceImplInterface() = default;
  virtual ~PDOOCServiceImplInterface() = default;

  virtual void decode_recv_new_requests(const proto::DisaggRequests* request,
                                        proto::DisaggResponses* response) {}

  virtual void decode_recv_first_generation(
      const proto::DisaggGenerations* request,
      proto::Status* response) {}

  virtual void decode_recv_multi_generations(
      const proto::MultiGenerationsRequests* request,
      proto::Status* response) {}

  virtual bool prefill_recv_generation(
      const proto::DisaggStreamGeneration* request,
      proto::Status* response) {
    return true;
  }

  virtual void prefill_recv_generations(
      const proto::DisaggStreamGenerations* requests,
      proto::StatusSet* responses) {}

  virtual void prefill_recv_pull_signal(const proto::PullSignal* request,
                                        proto::Status* response) {}
};

class PDOOCServiceImpl final : public PDOOCServiceImplInterface {
 public:
  explicit PDOOCServiceImpl(PDOOCScheduler* scheduler, Engine* engine);
  ~PDOOCServiceImpl() = default;

  bool prefill_recv_generation(const proto::DisaggStreamGeneration* request,
                               proto::Status* response) override;

  void prefill_recv_generations(const proto::DisaggStreamGenerations* requests,
                                proto::StatusSet* responses) override;

  void decode_recv_new_requests(const proto::DisaggRequests* request,
                                proto::DisaggResponses* response) override;

  void decode_recv_first_generation(const proto::DisaggGenerations* request,
                                    proto::Status* response) override;

  void decode_recv_multi_generations(
      const proto::MultiGenerationsRequests* request,
      proto::Status* response) override;

  void prefill_recv_pull_signal(const proto::PullSignal* request,
                                proto::Status* response) override;

 private:
  std::shared_ptr<Request> generate_request(const proto::DisaggRequest& req);

  PDOOCScheduler* scheduler_;  // not owned
  Engine* engine_;             // not owned
};

}  // namespace xllm
