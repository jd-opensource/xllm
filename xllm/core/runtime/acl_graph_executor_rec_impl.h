/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================*/

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

// ACL graph executor for multi-step / beam-search decode.
// This implementation is separated from the default AclGraphExecutorImpl
// to keep the control flow simple and avoid too many feature flags in one
// class. It is selected at runtime in Executor based on
// FLAGS_enable_beam_search_kernel.
class AclGraphRec {
 public:
  AclGraphRec() = default;

  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_size);

  torch::Tensor replay(const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params);

  torch::Tensor get_hidden_states() const { return hidden_states_; }

  torch::Tensor get_hidden_states(uint32_t actual_batch_size) const {
    return hidden_states_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
  }

 private:
  void copy_data_to_graph_buffer(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 const ModelInputParams& params,
                                 uint32_t actual_batch_size);

  void print_graph_tensors() const;

  c10_npu::NPUGraph graph_;
  uint32_t batch_size_{0};

  torch::Tensor flatten_tokens_;
  torch::Tensor flatten_positions_;
  torch::Tensor new_cache_slots_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor block_tables_;
  torch::Tensor hidden_states_;
  torch::Tensor graph_buffer_;
};

class AclGraphRecExecutorImpl : public ExecutorImpl {
 public:
  AclGraphRecExecutorImpl(CausalLM* model,
                          const ModelArgs& args,
                          const torch::Device& device,
                          const runtime::Options& options);

  ~AclGraphRecExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for multi-step / beam-search decode.
  // tokens: [num_decode_tokens]
  // positions: [num_decode_tokens]
  // returns: [num_decode_tokens, hidden_size]
  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    std::vector<KVCache>& kv_caches,
                    const ModelInputParams& params) override;

 private:
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  absl::flat_hash_map<uint32_t, std::unique_ptr<AclGraphRec>> graphs_;

  uint32_t get_bucket_size(uint32_t batch_size) const;
};

}  // namespace xllm
