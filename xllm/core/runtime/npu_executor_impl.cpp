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

#include "npu_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm {

NpuExecutorImpl::NpuExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {}

ForwardInput NpuExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// tokens: [num_tokens]
// positions: [num_tokens] token pos in the sequence
// returns: [num_tokens, hidden_size]
torch::Tensor NpuExecutorImpl::run(
    const std::vector<torch::Tensor>& tokens,
    const std::vector<torch::Tensor>& positions,
    std::vector<KVCache>& kv_caches,
    const std::vector<ModelInputParams>& params) {
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm
