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

#include <memory>

#include "common/macros.h"
#include "page_manager.h"
#include "page_manager.pb.h"

namespace xllm {
class PageManagerService : public proto::DistributePageManager {
 public:
  PageManagerService(int32_t global_rank,
                     int32_t world_size,
                     const torch::Device& d);
  ~PageManagerService() = default;

  void set_page_manager(std::unique_ptr<PageManager> page_manager);

  // service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void Allocate(::google::protobuf::RpcController* controller,
                const proto::AllocatePagesRequest* request,
                proto::Status* response,
                ::google::protobuf::Closure* done) override;

  void Deallocate(::google::protobuf::RpcController* controller,
                  const proto::SeqId* request,
                  proto::Empty* response,
                  ::google::protobuf::Closure* done) override;

  void Cache(::google::protobuf::RpcController* controller,
             const proto::SeqId* request,
             proto::Empty* response,
             ::google::protobuf::Closure* done) override;

  void NumFreePagesPerLayer(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::NumPages* response,
                            ::google::protobuf::Closure* done) override;

  void NumUsedPagesPerLayer(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::NumPages* response,
                            ::google::protobuf::Closure* done) override;

  void KvCacheUtilization(::google::protobuf::RpcController* controller,
                          const proto::Empty* request,
                          proto::Utilization* response,
                          ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerService);

 private:
  bool initialized_;
  int32_t global_rank_;
  int32_t world_size_;
  torch::Device device_;
  ThreadPool threadpool_{5};
  std::unique_ptr<PageManager> page_manager_;
};

}  // namespace xllm