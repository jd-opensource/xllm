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

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/mm_data.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "rec_batch_input_builder.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

class OneRecBatchInputBuilderCache final {
 public:
  OneRecBatchInputBuilderCache() = default;

  OneRecBatchInputBuilderCache(const OneRecBatchInputBuilderCache&) = delete;
  OneRecBatchInputBuilderCache& operator=(const OneRecBatchInputBuilderCache&) =
      delete;
  OneRecBatchInputBuilderCache(OneRecBatchInputBuilderCache&&) = delete;
  OneRecBatchInputBuilderCache& operator=(OneRecBatchInputBuilderCache&&) =
      delete;

 private:
  friend class OneRecBatchInputBuilder;

  class MemoryPool final {
   public:
    std::vector<int32_t>& get_int32_vector(size_t reserve_size = 0) {
      if (pool_index_ >= int32_pools_.size()) {
        int32_pools_.emplace_back();
      }
      std::vector<int32_t>& vec = int32_pools_[pool_index_++];
      vec.clear();
      if (reserve_size > 0) {
        vec.reserve(reserve_size);
      }
      return vec;
    }

    void reset() { pool_index_ = 0; }

   private:
    std::vector<std::vector<int32_t>> int32_pools_;
    size_t pool_index_ = 0;
  };

  struct CacheData {
    std::vector<int32_t> encoder_tokens;
    std::vector<int32_t> encoder_seq_lens;
    std::vector<torch::Tensor> encoder_sparse_embeddings;
    std::vector<torch::Tensor> decoder_context_embeddings;
  };

  void ensure_tensors_initialized() {
    if (!tensors_initialized_) {
      fixed_positions_tensor_ = torch::tensor({0}, torch::kInt);
      fixed_encoder_positions_tensor_ = torch::tensor({0}, torch::kInt);
      empty_tensor_ = torch::tensor(std::vector<int32_t>{}, torch::kInt);
      tensors_initialized_ = true;
    }
  }

  MemoryPool memory_pool_;
  CacheData cache_data_;
  torch::Tensor fixed_positions_tensor_;
  torch::Tensor fixed_encoder_positions_tensor_;
  torch::Tensor empty_tensor_;
  bool tensors_initialized_ = false;
};

class OneRecBatchInputBuilder : public RecBatchInputBuilder {
 public:
  explicit OneRecBatchInputBuilder(
      const std::vector<SequencesGroup*>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      const uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      ThreadPool* thread_pool,
      OneRecBatchInputBuilderCache* perf_cache);

 public:
  ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  const std::vector<SequencesGroup*>& sequence_groups_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  const std::vector<MMData>& mm_data_vec_;
  std::vector<BlockTransferInfo>* swap_block_transfer_infos_ = nullptr;
  const uint64_t batch_id_;
  const ModelArgs* args_ = nullptr;
  ThreadPool* thread_pool_ = nullptr;
  BatchForwardType batch_forward_type_;
  OneRecBatchInputBuilderCache* perf_cache_ = nullptr;
};

}  // namespace xllm
