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

#include "parallel_state.h"

#include "core/util/utils.h"

#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "npu_process_group.h"
#endif

namespace {

torch::Tensor remove_paddings_after_all_gather(
    const torch::Tensor& input,
    int64_t padding_to_token_num,
    const std::vector<int>& token_num_list) {
  std::vector<torch::Tensor> group_tensors;
  int64_t offset = 0;
  for (const auto& token_num : token_num_list) {
    if (token_num != 0) {
      auto tensor_slice = input.slice(0, offset, offset + token_num);
      group_tensors.push_back(tensor_slice);
    }
    offset += padding_to_token_num;
  }
  if (group_tensors.size() == 1) {
    return group_tensors[0];
  }

  return torch::cat(group_tensors).contiguous();
}

}  // namespace

namespace xllm {
namespace parallel_state {

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  const int32_t rank = process_group->rank();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/dim).contiguous();
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  if (world_size == 1) {
    return input;
  }
  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;

  const bool num_tokens_equal =
      std::all_of(token_num_list.begin(),
                  token_num_list.end(),
                  [first_token_num = token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return gather(input, process_group, 0);
  }

  int32_t max_num_tokens = xllm::util::max(token_num_list);
  int32_t num_padding = max_num_tokens - token_num_list[rank];
  auto padded_input = input;
  if (num_padding > 0) {
    std::vector<int64_t> pad = {0, 0, 0, num_padding};
    padded_input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(pad));
  }

  auto gathered_input = gather(padded_input, process_group, 0);
  return remove_paddings_after_all_gather(
      gathered_input, max_num_tokens, token_num_list);
}

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  if (world_size == 1) {
    return input;
  }

  std::vector<torch::Tensor> gathered_tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    gathered_tensors[i] = torch::empty_like(input);
  }
  process_group->allgather(input, gathered_tensors);

  int32_t dim = -1;
  size_t num_chunks = 3;
  std::vector<torch::Tensor> ordered_tensors;
  int64_t shard_size = input.size(dim) / num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    for (size_t j = 0; j < world_size; ++j) {
      auto shard_tensor =
          gathered_tensors[j].slice(dim, shard_size * i, shard_size * (i + 1));
      ordered_tensors.push_back(shard_tensor);
    }
  }
  return torch::cat(ordered_tensors, dim).contiguous();
}

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }
  process_group->allreduce(input);
  return input;
}

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  // get the size for last dimension
  const int32_t dim_size = input.size(dim);
  CHECK(dim_size % world_size == 0)
      << "dim_size " << dim_size << " cannot be divided by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(dim_size / world_size, dim);
  const int32_t rank = process_group->rank();
  return tensor_list[rank];
}

std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices) {
#if defined(USE_NPU)
  CHECK(!devices.empty()) << "devices should not be empty";

  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }

  std::vector<HcclComm> comms(devices.size());
  const int world_size = static_cast<int>(devices.size());
  // HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(),comms.data()));

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupHCCL>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }

  return process_groups;
#else
  LOG(FATAL) << "non-NPU device is not supported";
#endif
}

}  // namespace parallel_state
}  // namespace xllm
