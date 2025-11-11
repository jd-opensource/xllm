// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H
#define OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H

#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "OperationCacheCompute.h"
#include "Utils.h"
#include "atb/atb_infer.h"

namespace atb {

template <typename ParamType>
class OpParamCache {
 public:
  static OpParamCache& getInstance();

  atb::Operation* getOperation(const ParamType& param, const std::string& name);
  atb::Operation* getOperation(uint64_t hash_id);
  void saveOperation(uint64_t hash_id, atb::Operation* op);

 private:
  OpParamCache();

  OpParamCache(const OpParamCache&) = delete;
  OpParamCache& operator=(const OpParamCache&) = delete;

  ~OpParamCache();

  std::unordered_map<uint64_t, atb::Operation*> op_map_;
  mutable std::mutex mutex_;
};

template <typename ParamType>
atb::Operation* CreateAtbOperation(const ParamType& param,
                                   const std::string& name) {
  atb::Operation* op = nullptr;
  atb::CreateOperation(param, &op);
  TORCH_CHECK(op != nullptr, name, " CreateOperation failed!");
  return op;
}

template <typename ParamType>
OpParamCache<ParamType>& OpParamCache<ParamType>::getInstance() {
  static OpParamCache instance;
  return instance;
}

template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::getOperation(const ParamType& param,
                                                      const std::string& name) {
  const auto is_capturing =
      static_cast<int>(c10_npu::currentStreamCaptureStatusMayInitCtx());
  if (is_capturing) {
    // The atb operator does not support operator reuse, when operator creation
    // and execution in separate threads.
    return CreateAtbOperation(param, name);
  } else {
    uint64_t hashValue = computeHash(param);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto op_cache = op_map_.find(hashValue);
      if (op_cache != op_map_.end()) {
        return op_cache->second;
      }
      atb::Operation* op = CreateAtbOperation(param, name);
      op_map_[hashValue] = op;
      return op;
    }
  }
}

template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::getOperation(uint64_t hash_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto op_cache = op_map_.find(hash_id);
  if (op_cache != op_map_.end()) {
    return op_cache->second;
  }

  atb::Operation* op = nullptr;
  return op;
}

template <typename ParamType>
void OpParamCache<ParamType>::saveOperation(uint64_t hash_id,
                                            atb::Operation* op) {
  std::lock_guard<std::mutex> lock(mutex_);
  op_map_[hash_id] = op;
  return;
}

template <typename ParamType>
OpParamCache<ParamType>::OpParamCache() {
  // To satisfy the destructuring order, ContextManager should be instantiated
  // before OpParamCache.
  atb::utils::ContextManager::GetInstance();
}

template <typename ParamType>
OpParamCache<ParamType>::~OpParamCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& op_item : op_map_) {
    DestroyOperation(op_item.second);
  }
}

}  // namespace atb

#endif  // OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H
