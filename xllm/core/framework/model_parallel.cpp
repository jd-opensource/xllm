#include "model_parallel.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "model/model_args.h"

namespace llm {

torch::Tensor gather_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  const auto rank = parallel_args.rank();
  auto* process_group = parallel_args.process_group();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/-1).contiguous();
}

torch::Tensor reduce_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }
  auto* process_group = parallel_args.process_group();
  process_group->allreduce(input);
  return input;
}

torch::Tensor scatter_to_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  // get the size for last dimension
  const auto last_dim_size = input.size(-1);
  CHECK(last_dim_size % world_size == 0)
      << "last_dim_size " << last_dim_size << " not divisible by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(last_dim_size / world_size, /*dim=*/-1);
  const auto rank = parallel_args.rank();
  return tensor_list[rank];
}

torch::Tensor gather_from_data_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  const auto rank = parallel_args.rank();
  auto* process_group = parallel_args.process_group();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/0).contiguous();
}

}  // namespace llm
