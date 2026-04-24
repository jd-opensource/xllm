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

#include "layers/mlu/deepseek_v4/indexer_v2.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>

#include "framework/parallel_state/parallel_state.h"

namespace {

bool is_power_of_two(int64_t n) { return n > 0 && ((n & (n - 1)) == 0); }

torch::Tensor create_hadamard_matrix(int64_t n,
                                     torch::Dtype dtype,
                                     torch::Device device,
                                     bool normalize) {
  CHECK(is_power_of_two(n)) << "hadamard_matrix: n must be a power of two.";

  const torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor h = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    torch::Tensor top = torch::cat({h, h}, /*dim=*/1);
    torch::Tensor bottom = torch::cat({h, -h}, /*dim=*/1);
    h = torch::cat({top, bottom}, /*dim=*/0);
  }

  if (normalize) {
    h = h / std::sqrt(static_cast<double>(n));
  }
  return h;
}

torch::Tensor hadamard_transform_ref(const torch::Tensor& x,
                                     const torch::Tensor& h_matrix,
                                     double scale) {
  const auto x_shape = x.sizes();
  const int64_t dim = x.size(-1);
  torch::Tensor x_2d = x.reshape({-1, dim});

  const double log_dim = std::ceil(std::log2(static_cast<double>(dim)));
  const int64_t dim_padded =
      static_cast<int64_t>(1ULL << static_cast<uint64_t>(log_dim));
  if (dim != dim_padded) {
    x_2d = torch::nn::functional::pad(
        x_2d,
        torch::nn::functional::PadFuncOptions({0, dim_padded - dim})
            .mode(torch::kConstant)
            .value(0));
  }

  torch::Tensor x_float = x_2d.to(torch::kFloat32);
  torch::Tensor h_float = h_matrix.to(torch::kFloat32);
  torch::Tensor out_full = torch::matmul(x_float, h_float.t());

  if (std::abs(scale - 1.0) > 1e-9) {
    out_full.mul_(scale);
  }

  torch::Tensor out = out_full.index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, dim)});
  return out.reshape(x_shape).to(x.dtype());
}

torch::Tensor rotate_activation(const torch::Tensor& input,
                                const torch::Tensor& hadamard_matrix) {
  CHECK(input.dtype() == torch::kBFloat16)
      << "rotate_activation: input must be bfloat16.";
  const int64_t hidden_size = input.size(-1);
  const double scale = std::pow(static_cast<double>(hidden_size), -0.5);
  return hadamard_transform_ref(input, hadamard_matrix, scale);
}

}  // namespace

namespace xllm {
namespace layer {

IndexerV2Impl::IndexerV2Impl(int64_t dim,
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
                             const torch::TensorOptions& options)
    : dim_(dim),
      n_heads_(index_n_heads),
      head_dim_(index_head_dim),
      rope_head_dim_(rope_head_dim),
      index_topk_(index_topk),
      q_lora_rank_(q_lora_rank),
      window_size_(window_size),
      compress_ratio_(compress_ratio),
      rotary_emb_(rotary_emb),
      int_opts_(torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(options.device())) {
  tp_size_ = parallel_args.world_size();
  tp_rank_ = parallel_args.rank();
  tp_group_ = parallel_args.tp_group_;

  n_local_heads_ = n_heads_ / tp_size_;
  CHECK(n_heads_ % tp_size_ == 0)
      << "Number of heads must be divisible by TP size";

  softmax_scale_ = std::pow(static_cast<double>(head_dim_), -0.5) *
                   std::pow(static_cast<double>(n_heads_), -0.5);

  QuantArgs quant_args;
  wq_b_ = register_module("wq_b",
                          ColumnParallelLinear(q_lora_rank_,
                                               n_heads_ * head_dim_,
                                               /*bias=*/false,
                                               /*gather_output=*/false,
                                               quant_args,
                                               tp_group_,
                                               options));

  weights_proj_ = register_module("weights_proj",
                                  ColumnParallelLinear(dim_,
                                                       n_heads_,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       quant_args,
                                                       tp_group_,
                                                       options));

  compressor_ = register_module("compressor",
                                Compressor(dim_,
                                           head_dim_,
                                           rope_head_dim_,
                                           compress_ratio_,
                                           cached_state_num,
                                           norm_eps,
                                           /*rotate=*/true,
                                           rotary_emb_,
                                           options));

  const double log_head_dim =
      std::ceil(std::log2(static_cast<double>(head_dim_)));
  const int64_t head_dim_padded =
      static_cast<int64_t>(1ULL << static_cast<uint64_t>(log_head_dim));
  hadamard_matrix_ = create_hadamard_matrix(
      head_dim_padded, torch::kFloat32, torch::kCPU, /*normalize=*/false);
  hadamard_matrix_ =
      hadamard_matrix_.to(options.device(), options.dtype().toScalarType());
}

torch::Tensor IndexerV2Impl::preprocess_indexer_q(
    const torch::Tensor& qr,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor q = wq_b_->forward(qr);
  q = q.view({q.size(0), n_local_heads_, head_dim_});

  torch::Tensor q_pe = q.slice(-1, head_dim_ - rope_head_dim_, head_dim_);
  rotary_emb_->forward(q_pe,
                       positions,
                       q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);

  q = rotate_activation(q, hadamard_matrix_);
  return q;
}

torch::Tensor IndexerV2Impl::compute_index_scores(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache_slice,
    const torch::Tensor& weights) {
  const int64_t seq_len = q.size(0);
  const int64_t n_heads = q.size(1);

  torch::Tensor q_flat = q.reshape({seq_len * n_heads, head_dim_});
  torch::Tensor scores = torch::mm(q_flat, kv_cache_slice.t());
  scores = scores.reshape({seq_len, n_heads, kv_cache_slice.size(0)});
  scores = scores.relu_() * weights.unsqueeze(-1);
  scores = scores.sum(/*dim=*/1);
  return scores;
}

torch::Tensor IndexerV2Impl::apply_causal_mask(const torch::Tensor& index_score,
                                               int64_t seqlen,
                                               int64_t num_compressed) {
  torch::Tensor range_kv = torch::arange(num_compressed, int_opts_);
  torch::Tensor range_q = torch::arange(1, seqlen + 1, int_opts_);
  torch::Tensor mask =
      range_kv.unsqueeze(0) >=
      torch::floor_divide(range_q, compress_ratio_).unsqueeze(1);
  return index_score +
         torch::where(mask,
                      torch::full({},
                                  -std::numeric_limits<float>::infinity(),
                                  index_score.options()),
                      torch::zeros({}, index_score.options()));
}

torch::Tensor IndexerV2Impl::process_sequence(
    const torch::Tensor& q_seq,
    const torch::Tensor& weights_seq,
    const torch::Tensor& kv_cache_slice,
    int64_t start_pos,
    int64_t seqlen,
    int64_t offset) {
  torch::Tensor index_score =
      compute_index_scores(q_seq, kv_cache_slice, weights_seq);

  if (tp_group_ != nullptr && tp_size_ > 1) {
    index_score = parallel_state::reduce(index_score, tp_group_);
  }

  const int64_t num_compressed = kv_cache_slice.size(0);
  if (start_pos == 0) {
    index_score = apply_causal_mask(index_score, seqlen, num_compressed);
  }

  const int64_t k = std::min(index_topk_, num_compressed);
  torch::Tensor topk_idxs = std::get<1>(index_score.topk(k, /*dim=*/-1));

  if (start_pos == 0) {
    torch::Tensor range_q =
        torch::arange(1, seqlen + 1, int_opts_).unsqueeze(1);
    torch::Tensor mask =
        topk_idxs >= torch::floor_divide(range_q, compress_ratio_);
    topk_idxs =
        torch::where(mask, torch::full_like(topk_idxs, -1), topk_idxs + offset);
  } else {
    topk_idxs = topk_idxs + offset;
  }

  return topk_idxs;
}

std::vector<torch::Tensor> IndexerV2Impl::forward(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const AttentionMetadata& attn_metadata,
    const std::vector<int64_t>& batch_to_kv_state,
    torch::Tensor& kv_cache,
    const torch::Tensor& freqs_cis) {
  const torch::Tensor& q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  const torch::Tensor& query_lens = attn_metadata.q_seq_lens;
  const torch::Tensor& seq_lens = attn_metadata.kv_seq_lens;
  const torch::Tensor& block_tables = attn_metadata.block_table;

  torch::Tensor start_positions = seq_lens - query_lens;
  const int64_t batch_size = start_positions.size(0);
  CHECK(batch_to_kv_state.size() == static_cast<size_t>(batch_size))
      << "In indexer layer, batch_to_kv_state len " << batch_to_kv_state.size()
      << " != start_positions len " << batch_size;

  torch::Tensor q_pack =
      preprocess_indexer_q(qr, q_cu_seq_lens, positions, attn_metadata);

  (void)compressor_->forward(x,
                             positions,
                             block_tables,
                             q_cu_seq_lens,
                             seq_lens,
                             batch_to_kv_state,
                             kv_cache,
                             /*window_offset=*/0,
                             freqs_cis);

  torch::Tensor weights_pack = weights_proj_->forward(x) * softmax_scale_;

  std::vector<torch::Tensor> topk_idxs_list;
  topk_idxs_list.reserve(static_cast<size_t>(batch_size));
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t query_start = q_cu_seq_lens[i].item<int64_t>();
    const int64_t query_end = q_cu_seq_lens[i + 1].item<int64_t>();
    const int64_t seqlen = query_end - query_start;
    const int64_t start_pos = start_positions[i].item<int64_t>();
    const int64_t end_pos = start_pos + seqlen;
    const int64_t offset = offsets[i].item<int64_t>();
    const int64_t num_compressed = end_pos / compress_ratio_;

    torch::Tensor q_seq = q_pack.slice(0, query_start, query_end);
    torch::Tensor weights_seq = weights_pack.slice(0, query_start, query_end);
    torch::Tensor block_indices =
        block_tables.index({i, torch::indexing::Slice(0, num_compressed)});
    torch::Tensor kv_cache_slice =
        kv_cache.index({block_indices}).reshape({-1, head_dim_});

    topk_idxs_list.push_back(process_sequence(
        q_seq, weights_seq, kv_cache_slice, start_pos, seqlen, offset));
  }

  return topk_idxs_list;
}

void IndexerV2Impl::load_state_dict(const StateDict& state_dict) {
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  compressor_->load_state_dict(state_dict.get_dict_with_prefix("compressor."));
}

}  // namespace layer
}  // namespace xllm
