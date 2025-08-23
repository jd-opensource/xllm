#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/torch_npu.h>
#endif
#include <boost/algorithm/string.hpp>
#include <memory>

#include "linear_impl.h"

namespace llm {
namespace {
#define MAKE_COLUMN_PARALLEL_LINEAR(LinearlImplClass) \
  std::make_shared<LinearlImplClass>(                 \
      in_features, out_features, bias, gather_output, parallel_args, options);

#define MAKE_ROW_PARALLEL_LINEAR(LinearlImplClass)          \
  std::make_shared<LinearlImplClass>(in_features,           \
                                     out_features,          \
                                     bias,                  \
                                     input_is_parallelized, \
                                     if_reduce_results,     \
                                     parallel_args,         \
                                     options);

std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  return MAKE_COLUMN_PARALLEL_LINEAR(ColumnParallelLinearImpl);
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool if_reduce_results,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  return MAKE_ROW_PARALLEL_LINEAR(RowParallelLinearImpl);
}
}  // namespace

// construct a ColumnParallelLinear.
// chose right implementation based on the args.
ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const QuantArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 quant_args,
                                                 parallel_args,
                                                 options)) {}

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 {}, /*quant_args*/
                                                 parallel_args,
                                                 options)) {}

// construct a rotary positional embedding.
// chose right implementation based on the args.
RowParallelLinear::RowParallelLinear(int64_t in_features,
                                     int64_t out_features,
                                     bool bias,
                                     bool input_is_parallelized,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : ModuleHolder(create_row_parallel_linear(in_features,
                                              out_features,
                                              bias,
                                              input_is_parallelized,
                                              /*if_reduce_results*/ true,
                                              quant_args,
                                              parallel_args,
                                              options)) {}

RowParallelLinear::RowParallelLinear(int64_t in_features,
                                     int64_t out_features,
                                     bool bias,
                                     bool input_is_parallelized,
                                     bool if_reduce_results,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : ModuleHolder(create_row_parallel_linear(in_features,
                                              out_features,
                                              bias,
                                              input_is_parallelized,
                                              if_reduce_results,
                                              quant_args,
                                              parallel_args,
                                              options)) {}

// construct a rotary positional embedding.
// chose right implementation based on the args.
ReplicatedLinear::ReplicatedLinear(int64_t in_features,
                                   int64_t out_features,
                                   bool bias,
                                   const QuantArgs& quant_args,
                                   const torch::TensorOptions& options)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 false,
                                                 quant_args,
                                                 ParallelArgs(0, 1, nullptr),
                                                 options)) {}

}  // namespace llm