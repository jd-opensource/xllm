#include "linear_impl.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/model_parallel.h"

namespace llm {

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : gather_output_(gather_output),
      parallel_args_(parallel_args),
      device_(options.device()) {
  const auto world_size = parallel_args_.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, in_features}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  input = input.to(device_);
  auto output = F::linear(input, weight_, bias_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  // call load_state_dict with identity transform
  load_state_dict(state_dict,
                  [](const torch::Tensor& tensor) { return tensor; });
}

void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict,
                                               TensorTransform transform_func) {
  CHECK(transform_func != nullptr) << "transform_func must be provided";
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 0
  LOAD_SHARDED_WEIGHT_WITH_TRANSFORM(weight, 0);

  if (bias_.defined()) {
    // load sharded bias on dim 0
    LOAD_SHARDED_WEIGHT_WITH_TRANSFORM(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load and merge the weights on dim 0
  LOAD_FUSED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    // load and merge the bias on dim 0
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool if_reduce_results,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : input_is_parallelized_(input_is_parallelized),
      if_reduce_results_(if_reduce_results),
      parallel_args_(parallel_args) {
  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features, in_features_per_partition}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }
  auto output = F::linear(input, weight_);
  if (if_reduce_results_ && parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  // N.B. need to apply bias after the reduce
  if (bias_.defined()) {
    output.add_(bias_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 1
  LOAD_SHARDED_WEIGHT(weight, 1);

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace llm
