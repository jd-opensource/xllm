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
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "page_manager.pb.h"
#include "page_manager_client.h"
#include "util/threadpool.h"

namespace xllm {

class RemotePageManager : public PageManagerClient {
 public:
  explicit RemotePageManager(int32_t global_rank,
                             const std::string& server_address,
                             const torch::Device& d);
  virtual ~RemotePageManager() = default;

  bool wait_for_server_ready(const std::string& server_address);

  bool allocate(int32_t& seq_id, size_t num_tokens);
  void deallocate(int32_t seq_id);
  void cache(int32_t seq_id);

  folly::SemiFuture<bool> allocate_async(int32_t& seq_id, size_t num_tokens);
  folly::SemiFuture<folly::Unit> deallocate_async(int32_t seq_id);
  folly::SemiFuture<folly::Unit> cache_async(int32_t seq_id);

  size_t num_free_pages_per_layer() const;
  size_t num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

  folly::SemiFuture<size_t> num_free_pages_per_layer_async();
  folly::SemiFuture<size_t> num_used_pages_per_layer_async();

 private:
  DISALLOW_COPY_AND_ASSIGN(RemotePageManager);

 private:
  int32_t global_rank_;

  // brpc connection resource
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::DistributePageManager_Stub> stub_;

  ThreadPool threadpool_;
  const torch::Device device_;
};
}  // namespace xllm