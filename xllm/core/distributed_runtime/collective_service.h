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

#include <absl/container/flat_hash_set.h>
#include <hccl/hccl.h>

#include <unordered_map>

#include "common/macros.h"
#include "worker.pb.h"

namespace xllm {

class CollectiveService : public proto::Collective {
 public:
  CollectiveService(int dp_group_num, int total_num, int device_idx);
  virtual ~CollectiveService() = default;

  void Sync(::google::protobuf::RpcController* controller,
            const proto::AddressInfo* request,
            proto::CommUniqueIdList* response,
            ::google::protobuf::Closure* done) override;

  // wait all worker connected
  std::unordered_map<int32_t, std::string> wait();

 private:
  DISALLOW_COPY_AND_ASSIGN(CollectiveService);

  void to_proto_list(const std::vector<HcclRootInfo>& src,
                     proto::CommUniqueIdList* dst);
  void from_proto_list(const proto::CommUniqueIdList& src,
                       std::vector<HcclRootInfo>* dst);

 private:
  int total_num_ = 0;
  std::vector<HcclRootInfo> root_infos_;
  std::mutex mutex_;
  std::unordered_map<int32_t, std::string> addrs_map_;
};

}  // namespace xllm
