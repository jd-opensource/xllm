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

#ifndef XLLM_CORE_KERNELS_NPU_ATB_OPERATION_CREATE_H
#define XLLM_CORE_KERNELS_NPU_ATB_OPERATION_CREATE_H

#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "atb/atb_infer.h"
#include "operation_cache_compute.h"
#include "utils.h"

namespace atb {

template <typename ParamType>
class OpParamCache {
 public:
  static OpParamCache& getInstance();

  atb::Operation* get_operation(const ParamType& param,
                                const std::string& name);
  atb::Operation* get_operation(uint64_t hash_id);
  void save_operation(uint64_t hash_id, atb::Operation* op);

 private:
  OpParamCache();

  OpParamCache(const OpParamCache&) = delete;
  OpParamCache& operator=(const OpParamCache&) = delete;

  ~OpParamCache();

  std::unordered_map<uint64_t, atb::Operation*> op_map_;
  mutable std::mutex mutex_;
};

template <typename ParamType>
atb::Operation* create_atb_operation(const ParamType& param,
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
atb::Operation* OpParamCache<ParamType>::get_operation(
    const ParamType& param,
    const std::string& name) {
  const auto is_capturing =
      static_cast<int>(c10_npu::currentStreamCaptureStatusMayInitCtx());
  if (is_capturing) {
    return create_atb_operation(param, name);
  } else {
    uint64_t hashValue = compute_hash(param);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto op_cache = op_map_.find(hashValue);
      if (op_cache != op_map_.end()) {
        return op_cache->second;
      }
      atb::Operation* op = create_atb_operation(param, name);
      op_map_[hashValue] = op;
      return op;
    }
  }
}

template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::get_operation(uint64_t hash_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto op_cache = op_map_.find(hash_id);
  if (op_cache != op_map_.end()) {
    return op_cache->second;
  }

  atb::Operation* op = nullptr;
  return op;
}

template <typename ParamType>
void OpParamCache<ParamType>::save_operation(uint64_t hash_id,
                                             atb::Operation* op) {
  std::lock_guard<std::mutex> lock(mutex_);
  op_map_[hash_id] = op;
  return;
}

template <typename ParamType>
OpParamCache<ParamType>::OpParamCache() {
  atb::utils::ContextManager::get_instance();
}

template <typename ParamType>
OpParamCache<ParamType>::~OpParamCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& op_item : op_map_) {
    DestroyOperation(op_item.second);
  }
}

}  // namespace atb

#endif  // XLLM_CORE_KERNELS_NPU_ATB_OPERATION_CREATE_H
