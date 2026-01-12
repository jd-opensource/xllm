/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/cuda/CUDAGraph.h>
#include <absl/container/flat_hash_map.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "options.h"

namespace xllm {

// Helper class to hold persistent parameters for CUDA graph execution
// Multiple CudaGraph instances can share the same CudaGraphPersistentParam
// object
class CudaGraphPersistentParam {
 public:
  CudaGraphPersistentParam(const ModelArgs& args,
                           const torch::Device& device,
                           const runtime::Options& options);

  ~CudaGraphPersistentParam() = default;

  // Update persistent tensors with new input data
  // If return_capture_params is true, returns a ModelInputParams with
  // persistent buffer references. padded_num_tokens must be > 0 when
  // return_capture_params is true, used for build new ModelInputParams for
  // capture. If return_capture_params is false, only updates persistent buffers
  // and returns std::nullopt.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens = 0,
                                         bool return_capture_params = false);

  // Getter methods for persistent tensors
  torch::Tensor persistent_tokens(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_tokens_;
  }
  torch::Tensor persistent_positions(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_positions_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_positions_;
  }
  torch::Tensor persistent_new_cache_slots(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_new_cache_slots_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_new_cache_slots_;
  }
  torch::Tensor persistent_block_tables(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_block_tables_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_block_tables_;
  }
  torch::Tensor hidden_states(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return hidden_states_;
  }
  // Setter for hidden_states (for assignment)
  void set_hidden_states(const torch::Tensor& value) {
    const uint32_t result_tokens = value.size(0);
    hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens)
        .copy_(value, /*non_blocking=*/true);
  }
  torch::Tensor q_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return q_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return q_seq_lens_;
  }
  torch::Tensor kv_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return kv_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return kv_seq_lens_;
  }
  torch::Tensor persistent_embedding(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_embedding_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_embedding_;
  }
  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_paged_kv_indptr_;
  }
  torch::Tensor persistent_paged_kv_indices(uint32_t actual_size) const {
    if (actual_size > 0) {
      return persistent_paged_kv_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_size);
    }
    return persistent_paged_kv_indices_;
  }
  torch::Tensor persistent_paged_kv_last_page_len(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_last_page_len_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_paged_kv_last_page_len_;
  }
  torch::Tensor persistent_decode_qo_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_decode_qo_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_decode_qo_indptr_;
  }

 private:
  const ModelArgs& args_;
  const torch::Device& device_;
  const runtime::Options& options_;

  // Persistent tensors - basic parameters
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_new_cache_slots_;
  torch::Tensor persistent_block_tables_;
  torch::Tensor hidden_states_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor persistent_embedding_;

  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr_;
  torch::Tensor persistent_paged_kv_indices_;
  torch::Tensor persistent_paged_kv_last_page_len_;
  torch::Tensor persistent_decode_qo_indptr_;

  // TODO maybe not used. or use q_cu_seq_lens instead.
  torch::Tensor persistent_chunked_prefill_qo_indptr_;
};

// CUDA graph executor using libtorch CUDAGraph for memory management
class CudaGraph {
 public:
  explicit CudaGraph(CudaGraphPersistentParam& persistent_param,
                     c10::DeviceIndex device_index)
      : persistent_param_(persistent_param), device_index_(device_index) {
    // Initialize capture stream in constructor
    initialize_capture_stream(device_index);
  }

  // Capture computation graph for given bucket num_tokens
  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens,
               const decltype(at::cuda::graph_pool_handle())& pool);

  // Replay captured graph with new input data
  torch::Tensor replay(const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       std::vector<KVCache>& kv_cache,
                       const ModelInputParams& params);

  // Get the hidden states from the last capture
  torch::Tensor get_hidden_states(uint32_t actual_num_tokens) const {
    return persistent_param_.hidden_states(actual_num_tokens);
  }

 private:
  // Print graph held tensors for debugging
  void print_graph_tensors() const;

  // Initialize capture stream if not already initialized
  void initialize_capture_stream(c10::DeviceIndex device_index);

  // CUDA graph for capturing and replaying
  at::cuda::CUDAGraph graph_;
  uint32_t padded_num_tokens_;

  // Reference to persistent parameters (shared across multiple CudaGraph
  // instances)
  CudaGraphPersistentParam& persistent_param_;

  // Cached capture stream, initialized on first capture
  std::optional<c10::cuda::CUDAStream> capture_stream_;
  c10::DeviceIndex device_index_;
};

// Executor implementation using CUDA graph optimization
class CudaGraphExecutorImpl : public ExecutorImpl {
 public:
  CudaGraphExecutorImpl(CausalLM* model,
                        const ModelArgs& args,
                        const torch::Device& device,
                        const runtime::Options& options);

  ~CudaGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    std::vector<KVCache>& kv_caches,
                    const ModelInputParams& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  // Lazy-loaded CUDA graphs for different num_tokens
  absl::flat_hash_map<uint32_t, std::unique_ptr<CudaGraph>> graphs_;

  // Persistent parameters shared across all CudaGraph instances
  std::unique_ptr<CudaGraphPersistentParam> persistent_param_;

  // CUDA graph memory pool shared across all CudaGraph instances
  decltype(at::cuda::graph_pool_handle()) graph_pool_;

  // Get bucket num_tokens for given num_tokens
  // For num_tokens < 8: use 1, 2, 4, 8
  // For num_tokens >= 8: use multiples of 8
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;
};
REGISTER_EXECUTOR("cuda", CudaGraphExecutorImpl);
}  // namespace xllm
