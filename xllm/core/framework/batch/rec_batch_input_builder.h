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

#include <future>
#include <vector>

#include "batch_input_builder.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/mm_data.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

class RecBatchInputBuilder : public BatchInputBuilder {
 public:
  explicit RecBatchInputBuilder(
      const std::vector<std::unique_ptr<SequencesGroup>>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
      const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
      std::vector<CacheBlockInfo>* swap_cache_block_infos,
      const ModelArgs* args,
      ThreadPool* thread_pool = nullptr);

 protected:
  // Provide protected access methods for subclasses - modified to access
  // parent's protected members
  const std::vector<std::unique_ptr<SequencesGroup>>& get_sequence_groups()
      const {
    return sequence_groups_;
  }
  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return allowed_max_tokens_;
  }
  const std::vector<torch::Tensor>& get_input_embeddings_vec() const {
    return input_embeddings_vec_;
  }
  const std::vector<MMData>& get_mm_data_vec() const { return mm_data_vec_; }
  const std::vector<CacheBlockInfo>* get_copy_in_cache_block_infos() const {
    return copy_in_cache_block_infos_;
  }
  const std::vector<CacheBlockInfo>* get_copy_out_cache_block_infos() const {
    return copy_out_cache_block_infos_;
  }
  std::vector<CacheBlockInfo>* get_swap_cache_block_infos() const {
    return swap_cache_block_infos_;
  }
  const ModelArgs* get_args() const { return args_; }
  ThreadPool* get_thread_pool() const { return thread_pool_; }

 public:
  // Main public interface
  ForwardInput build_rec_forward_input(uint32_t num_decoding_tokens,
                                       uint32_t min_decoding_batch_size);

 private:
  // Helper method to extract sequences from groups
  static std::vector<Sequence*> extract_sequences_from_groups(
      const std::vector<std::unique_ptr<SequencesGroup>>& sequence_groups);

  // Member variables - only keep sequence_groups_, others inherited from parent
  // class
  const std::vector<std::unique_ptr<SequencesGroup>>& sequence_groups_;

  // High performance cache system
  struct HighPerformanceCache {
    // Memory pool - avoid frequent allocation/deallocation
    struct MemoryPool {
      std::vector<std::vector<int32_t>> int32_pools;
      size_t pool_index = 0;

      std::vector<int32_t>& getInt32Vector(size_t reserve_size = 0) {
        if (pool_index >= int32_pools.size()) {
          int32_pools.emplace_back();
        }
        auto& vec = int32_pools[pool_index++];
        vec.clear();
        if (reserve_size > 0) vec.reserve(reserve_size);
        return vec;
      }

      void reset() { pool_index = 0; }
    };

    // Cache data structure
    struct CacheData {
      std::vector<int32_t> encoder_tokens;
      std::vector<int> encoder_seq_lens;
      std::vector<torch::Tensor> encoder_sparse_embeddings;
      std::vector<torch::Tensor> decoder_context_embeddings;
    };

    // Pre-created constant tensors
    torch::Tensor fixed_positions_tensor;
    torch::Tensor fixed_encoder_positions_tensor;
    torch::Tensor empty_tensor;

    MemoryPool memory_pool;
    CacheData cache_data;

    HighPerformanceCache() {
      // Pre-create commonly used tensors to avoid repeated creation
      fixed_positions_tensor = torch::tensor({0}, torch::kInt);
      fixed_encoder_positions_tensor = torch::tensor({0}, torch::kInt);
      empty_tensor = torch::tensor(std::vector<int32_t>{}, torch::kInt);
    }
  };

  static HighPerformanceCache perf_cache_;
};

}  // namespace xllm