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

#include <torch/torch.h>

#include <memory>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/deepseek_v4/compressor.h"

namespace xllm {
namespace layer {

// Improved Indexer implementation for DeepSeek V4 models.
// This version uses Compressor for KV compression and supports tensor
// parallelism for better scalability.
class IndexerV2Impl : public torch::nn::Module {
 public:
  IndexerV2Impl() = default;

  IndexerV2Impl(int64_t dim,
                int64_t index_n_heads,
                int64_t index_head_dim,
                int64_t rope_head_dim,
                int64_t index_topk,
                int64_t q_lora_rank,
                int64_t window_size,
                int64_t compress_ratio,
                int64_t cached_state_num,
                double norm_eps,
                std::shared_ptr<RotaryEmbeddingBase> rotary_emb,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const torch::Tensor& positions,
      const torch::Tensor& offsets,
      const AttentionMetadata& attn_metadata,
      const std::vector<int64_t>& batch_to_kv_state,
      torch::Tensor& kv_cache,
      const torch::Tensor& freqs_cis);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t dim_;
  int64_t n_heads_;
  int64_t n_local_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t index_topk_;
  int64_t q_lora_rank_;
  int64_t window_size_;
  int64_t compress_ratio_;
  double softmax_scale_;

  int64_t tp_size_;
  int64_t tp_rank_;
  ProcessGroup* tp_group_;

  ColumnParallelLinear wq_b_{nullptr};
  ColumnParallelLinear weights_proj_{nullptr};
  Compressor compressor_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
  torch::Tensor hadamard_matrix_;
  torch::TensorOptions int_opts_{};

  torch::Tensor preprocess_indexer_q(const torch::Tensor& qr,
                                     const torch::Tensor& q_cu_seq_lens,
                                     const torch::Tensor& positions,
                                     const AttentionMetadata& attn_metadata);

  torch::Tensor apply_causal_mask(const torch::Tensor& index_score,
                                  int64_t seqlen,
                                  int64_t num_compressed);

  torch::Tensor process_sequence(const torch::Tensor& q_seq,
                                 const torch::Tensor& weights_seq,
                                 const torch::Tensor& kv_cache_slice,
                                 int64_t start_pos,
                                 int64_t seqlen,
                                 int64_t offset);

  torch::Tensor compute_index_scores(const torch::Tensor& q,
                                     const torch::Tensor& kv_cache_slice,
                                     const torch::Tensor& weights);
};

TORCH_MODULE(IndexerV2);

}  // namespace layer
}  // namespace xllm
