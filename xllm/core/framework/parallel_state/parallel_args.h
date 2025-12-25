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

#include "core/common/macros.h"
#include "process_group.h"

#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "xllm_kernels/models/base/param/mapping.h"
#endif

#include <nlohmann/json.hpp>
#include <string>

namespace xllm {

struct ParallelArgs {
  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {}

  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               ProcessGroup* process_group,
               int32_t ep_size)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        process_group_(process_group),
        ep_size_(ep_size) {}

#if defined(USE_NPU)
  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               ProcessGroup* process_group,
               int32_t ep_size,
               nlohmann::json mapping_data,
               atb_speed::base::Mapping mapping,
               std::string dispatchAndCombinecommDomain,
               HcclComm dispatchAndCombineHcclComm)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        process_group_(process_group),
        ep_size_(ep_size),
        mapping_data_(mapping_data),
        mapping_(mapping),
        dispatchAndCombinecommDomain_(dispatchAndCombinecommDomain),
        dispatchAndCombineHcclComm_(dispatchAndCombineHcclComm) {}
#endif

  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               ProcessGroup* process_group)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        process_group_(process_group) {}

  ParallelArgs(int32_t rank,
               int32_t world_size,
               ProcessGroup* process_group,
               ProcessGroup* dp_local_process_group,
               int32_t dp_size)
      : rank_(rank),
        world_size_(world_size),
        process_group_(process_group),
        dp_local_process_group_(dp_local_process_group),
        dp_size_(dp_size) {}

  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // world size
  PROPERTY(int32_t, world_size) = 0;

  // dp size
  PROPERTY(int32_t, dp_size) = 1;

  // ep size
  PROPERTY(int32_t, ep_size) = 1;

  // atb hccl mapping json data
  PROPERTY(nlohmann::json, mapping_data);

#if defined(USE_NPU)
  // atb hccl mapping
  PROPERTY(atb_speed::base::Mapping, mapping);

  // atb hccl dispatchAndCombinecommDomain
  PROPERTY(std::string, dispatchAndCombinecommDomain);

  // atb hccl dispatchAndCombineHcclComm
  PROPERTY(HcclComm, dispatchAndCombineHcclComm);
#endif

  // the following pointers are unique pointers from CollectiveCommunicator
  //  So they are not owned by ParallelArgs.
  ProcessGroup* process_group_ = nullptr;
  ProcessGroup* dp_local_process_group_ = nullptr;
  ProcessGroup* tp_group_ = nullptr;
  ProcessGroup* moe_ep_group_ = nullptr;
  ProcessGroup* moe_tp_group_ = nullptr;
};

}  // namespace xllm