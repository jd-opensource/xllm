#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/torch_npu.h>
#endif
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {

using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;

// an interface for parallel linear layer.
// all linear classes should inherit from this class and implement the forward
// function.
class ParallelLinearImpl : public torch::nn::Module {
 public:
  ~ParallelLinearImpl() override = default;

  virtual torch::Tensor forward(torch::Tensor input) = 0;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights(const std::string& prefix = "") const = 0;

  // load state dict with a transform function
  virtual void load_state_dict(const StateDict& /*state_dict*/,
                               TensorTransform /*transform_func*/) {
    LOG(FATAL) << "not implemented";
  }

  // special load_state_dict for fused cases
  virtual void load_state_dict(const StateDict& /*state_dict*/,
                               const std::vector<std::string>& /*prefixes*/) {
    LOG(FATAL) << "not implemented";
  }
};

class ColumnParallelLinear
    : public torch::nn::ModuleHolder<ParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ParallelLinearImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options);

  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options);
};

class RowParallelLinear : public torch::nn::ModuleHolder<ParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ParallelLinearImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool bias,
                    bool input_is_parallelized,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);

  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool bias,
                    bool input_is_parallelized,
                    bool if_reduce_results,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);
};

class ReplicatedLinear : public torch::nn::ModuleHolder<ParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ParallelLinearImpl;
  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  ReplicatedLinear(int64_t in_features,
                   int64_t out_features,
                   bool bias,
                   const QuantArgs& quant_args,
                   const torch::TensorOptions& options);
};
}  // namespace xllm
