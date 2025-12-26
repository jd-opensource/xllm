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
#include <torch_mlu/csrc/framework/graphs/MLUGraph.h>

#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "options.h"

namespace xllm {
// Helper class to hold persistent parameters for graph execution
// Multiple MluGraph instances can share the same GraphPersistentParam object
class GraphPersistentParam {
 public:
  GraphPersistentParam(const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~GraphPersistentParam() = default;

  void init_params(const ModelInputParams& params,
                   uint32_t padding_num_tokens,
                   uint32_t padding_needed);

  // Update persistent tensors with new input data
  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelInputParams& params,
                           uint32_t padding_needed);

  // input tensors
  torch::Tensor tokens_;
  torch::Tensor positions_;
  ModelInputParams params_;
  // mrope
  bool use_mrope_ = false;
  torch::Tensor mrope_cos_;
  torch::Tensor mrope_sin_;
  // output
  torch::Tensor output_;

 private:
  // attn_metadata
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor new_cache_slots_;
  torch::Tensor block_table_;
  uint32_t num_decoding_tokens_;

  // for vl
  torch::Tensor input_embeds_;
  torch::Tensor pixel_values_;
  torch::Tensor image_grid_thw_;

  // for mtp model
  torch::Tensor embedding_;
};

// graph executor using libtorch MLUGraph for memory management
// MLUGraph provides mempool to manage temporary tensors during forward pass
class MluGraph {
 public:
  MluGraph(GraphPersistentParam* persistent_param, uint32_t padding_num_tokens);

  // Capture computation graph for given bucket num_tokens
  void capture(CausalLM* model,
               std::vector<KVCache>& kv_cache,
               torch_mlu::MempoolId_t& pool);

  // Replay captured graph with new input data
  void replay();
  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelInputParams& params,
                           bool is_init = false);

 private:
  // MLUGraph with mempool for managing temporary tensors during forward pass
  torch_mlu::MLUGraph graph_;

  // Reference to persistent parameters (shared across multiple MluGraph
  // instances)
  GraphPersistentParam* persistent_param_;  // not owned
  uint32_t padding_num_tokens_;
};

// Executor implementation using MLU graph optimization
// Uses MLUGraph mempool to reduce memory allocation overhead during inference
class MluGraphExecutorImpl : public ExecutorImpl {
 public:
  MluGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~MluGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    std::vector<KVCache>& kv_caches,
                    const ModelInputParams& params) override;

 private:
  CausalLM* model_;  // not owned

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  torch_mlu::MempoolId_t pool_;

  std::unordered_map<uint32_t, std::unique_ptr<MluGraph>> graphs_;
  std::unique_ptr<GraphPersistentParam> persistent_param_;
};
REGISTER_EXECUTOR("mlu", MluGraphExecutorImpl);
}  // namespace xllm
