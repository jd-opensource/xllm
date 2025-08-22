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

#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"
#include "worker.pb.h"

namespace xllm {

void proto_to_forward_input(const proto::ForwardInput* pb_forward_input,
                            ForwardInput& forward_inputs,
                            int64_t num_decoding_tokens);

void forward_input_to_proto(const RawForwardInput& inputs,
                            proto::ForwardInput* pb_forward_input);

void proto_to_forward_output(const proto::ForwardOutput& pb_output,
                             RawForwardOutput& raw_forward_output);

void forward_output_to_proto(const torch::Tensor& next_tokens,
                             const torch::Tensor& logprobs,
                             const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             const torch::Tensor& embeddings,
                             const torch::Tensor& expert_load_data,
                             int32_t prepared_layer_id,
                             proto::ForwardOutput* pb_forward_output);

Token build_token(int64_t index,
                  torch::Tensor token_ids,
                  torch::Tensor logprobs,
                  torch::Tensor top_tokens,
                  torch::Tensor top_logprobs);

void proto_to_cache_block_info(
    const proto::CacheBlockInfos& cache_block_info_pb,
    std::vector<CacheBlockInfo>& cache_block_info);

bool cache_block_info_to_proto(
    const std::vector<CacheBlockInfo>& cache_block_info,
    proto::CacheBlockInfos* cache_block_info_pb);

}  // namespace xllm
