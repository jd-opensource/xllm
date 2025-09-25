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

#include <absl/container/flat_hash_map.h>
#include <acl/acl.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "executor_impl.h"
#include "options.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

namespace xllm {

// ACL graph executor using libtorch NPUGraph for memory management
// NPUGraph provides mempool to manage temporary tensors during forward pass
class AclGraph {
 public:
  AclGraph() = default;

  // Capture computation graph for given bucket size
  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_size);

  // Replay captured graph with new input data
  torch::Tensor replay(const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params);

  // Get the hidden states from the last capture
  torch::Tensor get_hidden_states() const { return hidden_states_; }

  // Get the hidden states for actual batch size (slice to avoid padded data)
  torch::Tensor get_hidden_states(uint32_t actual_batch_size) const {
    return hidden_states_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
  }

 private:
  // Copy data to graph persistent buffers
  void copy_data_to_graph_buffer(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 const ModelInputParams& params,
                                 uint32_t actual_batch_size);

  // Print graph held tensors for debugging
  void print_graph_tensors() const;

  // NPUGraph with mempool for managing temporary tensors during forward pass
  c10_npu::NPUGraph graph_;
  uint32_t batch_size_;

  // Persistent tensors captured in graph for reuse
  torch::Tensor flatten_tokens_;
  torch::Tensor flatten_positions_;
  torch::Tensor new_cache_slots_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor block_tables_;
  torch::Tensor hidden_states_;
  torch::Tensor graph_buffer_;
};

// Executor implementation using ACL graph optimization
// Uses NPUGraph mempool to reduce memory allocation overhead during inference
class AclGraphExecutorImpl : public ExecutorImpl {
 public:
  AclGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~AclGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  torch::Tensor run(const std::vector<torch::Tensor>& tokens,
                    const std::vector<torch::Tensor>& positions,
                    std::vector<KVCache>& kv_caches,
                    const std::vector<ModelInputParams>& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  // Lazy-loaded ACL graphs for different batch sizes
  absl::flat_hash_map<uint32_t, std::unique_ptr<AclGraph>> graphs_;

  // Get bucket size for given batch size
  // For batch_size < 8: use 1, 2, 4, 8
  // For batch_size >= 8: use multiples of 8
  uint32_t get_bucket_size(uint32_t batch_size) const;
};

}  // namespace xllm