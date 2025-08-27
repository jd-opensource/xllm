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

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <vector>

#include "core/common/macros.h"
#include "mapping_npu.h"
#if defined(USE_NPU)
#include <hccl/hccl_types.h>

#include "hccl/hccl.h"
#include "xllm_kernels/models/base/param/mapping.h"
#endif

namespace xllm {

class ParallelArgs;
namespace parallel_state {

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args);

torch::Tensor gather(torch::Tensor input, const ParallelArgs& parallel_args);

torch::Tensor reduce(torch::Tensor input, const ParallelArgs& parallel_args);

torch::Tensor scatter(torch::Tensor input, const ParallelArgs& parallel_args);

}  // namespace parallel_state

class ProcessGroup;
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

  ProcessGroup* process_group_ = nullptr;
  ProcessGroup* dp_local_process_group_ = nullptr;

  // dp size
  PROPERTY(int32_t, dp_size) = 1;

  // ep size
  PROPERTY(int32_t, ep_size) = 1;

#if defined(USE_NPU)
  // atb hccl mapping json data
  PROPERTY(nlohmann::json, mapping_data);

  // atb hccl mapping
  PROPERTY(atb_speed::base::Mapping, mapping);

  // atb hccl dispatchAndCombinecommDomain
  PROPERTY(std::string, dispatchAndCombinecommDomain);

  // atb hccl dispatchAndCombineHcclComm
  PROPERTY(HcclComm, dispatchAndCombineHcclComm);
#endif
};

class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, const torch::Device& device)
      : rank_(rank), world_size_(world_size), device_(device) {}

  virtual ~ProcessGroup() = default;

  int rank() const { return rank_; }

  int world_size() const { return world_size_; }

  const torch::Device& device() const { return device_; }

  // allreduce: reduce the input tensor across all processes, and all processes
  // get the result.
  virtual void allreduce(torch::Tensor& input) = 0;

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(torch::Tensor input,
                         std::vector<torch::Tensor>& outputs) = 0;

  // Create a process group where each process has a single GPU
  // devices: list of devices to create process groups on.
  static std::vector<std::unique_ptr<ProcessGroup>> create_process_groups(
      const std::vector<torch::Device>& devices);

 private:
  // rank of current process.
  int rank_ = 0;

  // number of processes.
  int world_size_ = 0;

  // device of current process.
  torch::Device device_;
};

#if defined(USE_NPU)
class ProcessGroupHCCL : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupHCCL(int rank,
                   int world_size,
                   const torch::Device& device,
                   HcclComm comm);

  // Destructor.
  ~ProcessGroupHCCL() override;

  void allreduce(torch::Tensor& input) override;

  void allgather(torch::Tensor input,
                 std::vector<torch::Tensor>& outputs) override;

 private:
  HcclComm comm_ = nullptr;
};
#endif

}  // namespace xllm
