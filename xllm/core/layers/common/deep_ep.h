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

#include <torch/torch.h>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {
namespace layer {

// Buffer struct used to store communication tensors for DeepEP
struct DeepEPBuffer {
  torch::Tensor dispatch_send_token_tensor;
  torch::Tensor dispatch_recv_token_tensor;
  torch::Tensor combine_send_token_tensor;
  torch::Tensor combine_recv_token_tensor;
};

// DeepEPImpl manages distributed dispatch and combine operations for tokens
// between expert ranks
class DeepEPImpl : public torch::nn::Module {
 public:
  DeepEPImpl() = default;
  /**
   * @param dispatch_token_size Number of bytes in one token for the dispatch
   * stage
   * @param combine_token_size Number of bytes in one token for the combine
   * stage
   * @param max_num_tokens_per_rank Maximum number of tokens per rank
   * @param num_global_experts Number of global experts
   * @param rank Current process rank
   * @param nranks Total number of processes
   * @param parallel_args Parallel arguments
   * @param options Tensor options
   */
  DeepEPImpl(int64_t dispatch_token_size,
             int64_t combine_token_size,
             int64_t max_num_tokens_per_rank,
             int64_t num_global_experts,
             const ParallelArgs& parallel_args,
             const torch::TensorOptions& options);

  ~DeepEPImpl();

  // Dispatch tokens to other experts using all-to-all communication
  void dispatch(int64_t token_byte,
                int64_t token_num,
                const torch::Tensor& send_layout,
                const torch::Tensor& send_token_num,
                const torch::Tensor& recv_layout,
                const torch::Tensor& recv_token_num,
                const std::optional<torch::Tensor>& send_token = std::nullopt,
                const std::optional<torch::Tensor>& recv_token = std::nullopt);

  // Combine tokens from other experts using all-to-all communication
  void combine(int64_t token_byte,
               int64_t token_num,
               const torch::Tensor& send_src_layout,
               const torch::Tensor& send_dst_layout,
               const std::optional<torch::Tensor>& send_token = std::nullopt,
               const std::optional<torch::Tensor>& recv_token = std::nullopt);

  // Utility function to get the communication buffer
  DeepEPBuffer get_buffer() const;

 private:
  int64_t handle_ = 0;  // Communication handle
  int64_t max_num_tokens_per_rank_;
  bool is_initialized_ = false;
  const ParallelArgs& parallel_args_;
  const torch::TensorOptions& options_;

  // Buffers created for communication; registered to support module
  // state/save/load and .to(device)
  torch::Tensor dispatch_send_token_tensor_;
  torch::Tensor dispatch_recv_token_tensor_;
  torch::Tensor combine_send_token_tensor_;
  torch::Tensor combine_recv_token_tensor_;
};
TORCH_MODULE(DeepEP);

}  // namespace layer
}  // namespace xllm