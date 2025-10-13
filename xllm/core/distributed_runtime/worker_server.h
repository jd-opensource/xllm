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

#include <thread>

#include "common/macros.h"
#include "distributed_runtime/worker_service.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "runtime/executor.h"
#include "runtime/forward_params.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"
#include "worker.pb.h"

namespace xllm {

class WorkerServer {
 public:
  WorkerServer(int local_worker_idx,
               const std::string& master_node_addr,
               std::atomic<bool>& done,
               const ParallelArgs& parallel_args,
               const torch::Device& d,
               const runtime::Options& options,
               WorkerType worker_type);

  virtual ~WorkerServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(WorkerServer);

  void create_server(const runtime::Options& options,
                     std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& d,
                     int world_sizse,
                     int global_rank,
                     int32_t dp_size,
                     int local_rank,
                     int32_t ep_size);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> worker_thread_;
};

}  // namespace xllm
