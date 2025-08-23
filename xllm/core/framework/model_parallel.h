#pragma once

#include <torch/torch.h>

#include "parallel_state.h"

namespace llm {

torch::Tensor gather_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor reduce_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor scatter_to_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor gather_from_data_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

}  // namespace llm
