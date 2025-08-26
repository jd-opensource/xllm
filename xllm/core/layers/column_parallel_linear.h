#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_column_parallel_linear_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class ColumnParallelLinear
    : public torch::nn::ModuleHolder<NpuColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<NpuColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuColumnParallelLinearImpl;

  ColumnParallelLinear(const Context& context)
      : ModuleHolder(std::make_shared<NpuColumnParallelLinearImpl>(context)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm