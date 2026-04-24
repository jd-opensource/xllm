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

#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"

#include <glog/logging.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "core/kernels/ops_api.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace {

double find_correction_dim(double num_rotations,
                           int64_t dim,
                           double base,
                           int64_t max_seq_len) {
  const double pi = std::acos(-1.0);
  return dim * std::log(max_seq_len / (num_rotations * 2.0 * pi)) /
         (2.0 * std::log(base));
}

std::pair<int64_t, int64_t> find_correction_range(double low_rot,
                                                  double high_rot,
                                                  int64_t dim,
                                                  double base,
                                                  int64_t max_seq_len) {
  const int64_t low = std::max(
      static_cast<int64_t>(
          std::floor(find_correction_dim(low_rot, dim, base, max_seq_len))),
      int64_t{0});
  const int64_t high = std::min(
      static_cast<int64_t>(
          std::ceil(find_correction_dim(high_rot, dim, base, max_seq_len))),
      dim - 1);
  return {low, high};
}

torch::Tensor linear_ramp_factor(int64_t min, int64_t max, int64_t dim) {
  const int64_t max_adj = (min == max) ? max + 1 : max;
  torch::Tensor linear_func =
      (torch::arange(dim, torch::kFloat32) - min) / (max_adj - min);
  return torch::clamp(linear_func, 0.0, 1.0);
}

torch::Tensor precompute_freqs_cis(int64_t dim,
                                   int64_t seqlen,
                                   int64_t original_seq_len,
                                   double base,
                                   double factor,
                                   double beta_fast,
                                   double beta_slow,
                                   torch::Device device,
                                   torch::ScalarType dtype) {
  CHECK(dim % 2 == 0) << "precompute_freqs_cis: dim must be even";

  torch::Tensor freqs =
      1.0 / torch::pow(base, torch::arange(0, dim, 2, torch::kFloat32) / dim);
  if (original_seq_len > 0) {
    const auto [low, high] = find_correction_range(
        beta_fast, beta_slow, dim, base, original_seq_len);
    torch::Tensor smooth = 1.0 - linear_ramp_factor(low, high, dim / 2);
    freqs = freqs / factor * (1.0 - smooth) + freqs * smooth;
  }

  torch::Tensor t = torch::arange(seqlen, torch::kFloat32);
  torch::Tensor freqs_outer = torch::outer(t, freqs);
  torch::Tensor ones = torch::ones_like(freqs_outer);
  torch::Tensor freqs_cis = torch::polar(ones, freqs_outer);
  return freqs_cis.to(device, dtype);
}

}  // namespace

namespace xllm {
namespace layer {

// Helper function to get window indices for a single sequence
static torch::Tensor get_window_topk_idxs_single(int64_t start_pos,
                                                 int64_t seqlen,
                                                 int64_t window_size,
                                                 torch::Device device) {
  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

  if (start_pos >= window_size - 1) {
    // Decode phase with full window: return [0, 1, ..., window_size-1]
    return torch::arange(window_size, opts).unsqueeze(0);
  } else if (start_pos > 0) {
    // Decode phase with partial window: pad with -1
    auto indices = torch::arange(start_pos + 1, opts);
    auto padded = torch::nn::functional::pad(
        indices,
        torch::nn::functional::PadFuncOptions({0, window_size - start_pos - 1})
            .value(-1));
    return padded.unsqueeze(0);
  } else {
    // Prefill phase: build causal window mask
    // For each query position i, valid KV positions are max(0, i-window+1) to i
    auto base = torch::arange(seqlen, opts).unsqueeze(1);
    int64_t k = std::min(seqlen, window_size);
    auto offset = torch::arange(k, opts);
    auto matrix = (base - window_size + 1).clamp_min(0) + offset;
    // Mask out future positions
    matrix = torch::where(matrix > base, torch::full({}, -1, opts), matrix);
    return matrix;
  }
}

std::vector<torch::Tensor> get_window_topk_idxs(
    int64_t window_size,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& seq_lens,
    torch::Device device) {
  auto query_lens = q_cu_seq_lens.slice(0, 1) - q_cu_seq_lens.slice(0, 0, -1);
  auto start_positions = seq_lens - query_lens;

  int64_t batch_size = start_positions.size(0);
  std::vector<torch::Tensor> topk_idxs_list;
  topk_idxs_list.reserve(batch_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t start_pos = start_positions[i].item<int64_t>();
    int64_t seqlen = (q_cu_seq_lens[i + 1] - q_cu_seq_lens[i]).item<int64_t>();
    topk_idxs_list.push_back(
        get_window_topk_idxs_single(start_pos, seqlen, window_size, device));
  }

  return topk_idxs_list;
}

// Helper function to get compress indices for a single sequence
static torch::Tensor get_compress_topk_idxs_single(int64_t start_pos,
                                                   int64_t seqlen,
                                                   int64_t compress_ratio,
                                                   int64_t offset,
                                                   torch::Device device) {
  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

  if (start_pos > 0) {
    // Decode phase: return all valid compressed positions
    int64_t num_compressed = (start_pos + 1) / compress_ratio;
    return torch::arange(num_compressed, opts).unsqueeze(0) + offset;
  } else {
    // Prefill phase: build causal compressed mask
    int64_t num_compressed = seqlen / compress_ratio;
    auto matrix = torch::arange(num_compressed, opts).repeat({seqlen, 1});
    auto query_positions = torch::arange(1, seqlen + 1, opts).unsqueeze(1);
    auto valid_boundary = query_positions.div(compress_ratio, "floor");
    auto mask = matrix >= valid_boundary;
    matrix = torch::where(mask, torch::full({}, -1, opts), matrix + offset);
    return matrix;
  }
}

std::vector<torch::Tensor> get_compress_topk_idxs(
    int64_t compress_ratio,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& offsets,
    torch::Device device) {
  auto query_lens = q_cu_seq_lens.slice(0, 1) - q_cu_seq_lens.slice(0, 0, -1);
  auto start_positions = seq_lens - query_lens;

  int64_t batch_size = start_positions.size(0);
  std::vector<torch::Tensor> topk_idxs_list;
  topk_idxs_list.reserve(batch_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t start_pos = start_positions[i].item<int64_t>();
    int64_t seqlen = (q_cu_seq_lens[i + 1] - q_cu_seq_lens[i]).item<int64_t>();
    int64_t offset = offsets[i].item<int64_t>();
    topk_idxs_list.push_back(get_compress_topk_idxs_single(
        start_pos, seqlen, compress_ratio, offset, device));
  }

  return topk_idxs_list;
}

DeepSeekV4AttentionImpl::DeepSeekV4AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int64_t layer_id,
    int64_t cached_state_num)
    : layer_id_(layer_id),
      hidden_size_(args.hidden_size()),
      num_heads_(args.n_heads()),
      head_dim_(args.head_dim()),
      q_lora_rank_(args.q_lora_rank()),
      rope_head_dim_(args.rotary_dim()),
      eps_(args.rms_norm_eps()) {
  // Get parameters from ModelArgs
  compress_ratio_ = args.rope_scaling() > 0 ? args.rope_scaling() : 0;
  window_size_ = args.sliding_window();
  o_lora_rank_ =
      args.kv_lora_rank() > 0 ? args.kv_lora_rank() : args.q_lora_rank();
  o_groups_ = args.n_group() > 0 ? args.n_group() : 1;
  max_model_len_ = args.max_position_embeddings();
  original_seq_len_ = args.rope_scaling_original_max_position_embeddings();

  // Tensor parallelism setup
  tp_size_ = parallel_args.world_size();
  tp_rank_ = parallel_args.rank();
  tp_group_ = parallel_args.tp_group_;

  CHECK_EQ(num_heads_ % tp_size_, 0)
      << "num_heads must be divisible by tensor parallel size";
  num_local_heads_ = num_heads_ / tp_size_;
  o_local_groups_ = o_groups_ / tp_size_;

  softmax_scale_ = std::pow(static_cast<float>(head_dim_), -0.5f);

  // Query projection with LoRA
  if (q_lora_rank_ > 0) {
    wq_a_ = register_module(
        "wq_a",
        ReplicatedLinear(
            hidden_size_, q_lora_rank_, /*bias=*/false, QuantArgs(), options));
    q_norm_ = register_module("q_norm", RMSNorm(q_lora_rank_, eps_, options));
    wq_b_ = register_module("wq_b",
                            ColumnParallelLinear(q_lora_rank_,
                                                 num_heads_ * head_dim_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 tp_group_,
                                                 options));
  }

  // KV projection
  wkv_ = register_module(
      "wkv",
      ReplicatedLinear(
          hidden_size_, head_dim_, /*bias=*/false, QuantArgs(), options));
  kv_norm_ = register_module("kv_norm", RMSNorm(head_dim_, eps_, options));

  // Output projection: choose between parallel and replicated based on TP
  use_parallel_o_proj_ = (tp_size_ <= o_groups_);
  if (use_parallel_o_proj_) {
    // TODO: QuantArgs is all empty here. Check this part later if smoothquant
    // w8a8 is supported.
    wo_a_col_ =
        register_module("wo_a",
                        ColumnParallelLinear(num_heads_ * head_dim_ / o_groups_,
                                             o_groups_ * o_lora_rank_,
                                             /*bias=*/false,
                                             /*gather_output=*/false,
                                             QuantArgs(),
                                             tp_group_,
                                             options));
    wo_b_row_ =
        register_module("wo_b",
                        RowParallelLinear(o_groups_ * o_lora_rank_,
                                          hidden_size_,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          /*reduce=*/false,
                                          quant_args,
                                          tp_group_,
                                          options));
  } else {
    wo_a_rep_ =
        register_module("wo_a",
                        ReplicatedLinear(num_heads_ * head_dim_ / o_groups_,
                                         o_groups_ * o_lora_rank_,
                                         /*bias=*/false,
                                         QuantArgs(),
                                         options));
    wo_b_rep_ = register_module("wo_b",
                                ReplicatedLinear(o_groups_ * o_lora_rank_,
                                                 hidden_size_,
                                                 /*bias=*/false,
                                                 quant_args,
                                                 options));
  }

  // Precompute frequency tensor for RoPE
  double rope_theta = args.rope_theta();
  double compress_rope_theta = args.rope_scaling_high_freq_factor() > 0.0f
                                   ? args.rope_scaling_high_freq_factor()
                                   : rope_theta;
  double rope_factor = args.rope_scaling_factor();
  double beta_fast = static_cast<double>(args.rope_scaling_beta_fast());
  double beta_slow = static_cast<double>(args.rope_scaling_beta_slow());
  const bool use_compress_rope = compress_ratio_ > 0;
  double base_theta = use_compress_rope ? compress_rope_theta : rope_theta;
  int64_t rope_original_seq_len =
      use_compress_rope ? original_seq_len_ : int64_t{0};
  freqs_cis_ = precompute_freqs_cis(rope_head_dim_,
                                    max_model_len_,
                                    rope_original_seq_len,
                                    base_theta,
                                    rope_factor,
                                    beta_fast,
                                    beta_slow,
                                    torch::kCPU,
                                    torch::kComplexFloat);

  ModelArgs rope_args = args;
  if (use_compress_rope) {
    rope_args.rope_scaling_rope_type() = "deepseek_yarn";
    rope_args.rope_theta() = static_cast<float>(compress_rope_theta);
    rope_args.rope_scaling_original_max_position_embeddings() =
        original_seq_len_;
  } else {
    rope_args.rope_scaling_rope_type() = "default";
    rope_args.rope_theta() = static_cast<float>(rope_theta);
    rope_args.rope_scaling_original_max_position_embeddings() = 0;
  }

  auto create_rotary_emb = [&](const std::string& name,
                               bool inverse,
                               const torch::TensorOptions& opts) {
    return register_module(name,
                           create_mla_rotary_embedding(rope_args,
                                                       rope_head_dim_,
                                                       max_model_len_,
                                                       /*interleaved=*/true,
                                                       opts,
                                                       inverse));
  };

  // ROPE embedding
  rotary_emb_ = create_rotary_emb("rotary_emb", /*inverse=*/false, options);
  output_rotary_emb_ = create_rotary_emb(
      "output_rotary_emb", /*inverse=*/true, options.dtype(torch::kFloat32));

  // Attention sink
  attn_sink_ = register_parameter(
      "attn_sink",
      torch::empty({num_local_heads_},
                   options.dtype(torch::kFloat32).requires_grad(false)));

  // Compressor module
  if (compress_ratio_ > 0) {
    // TODO: rotary_emb_ needs proper initialization for compressor
    // Currently passing a null reference, will be refactored
    compressor_ =
        register_module("compressor",
                        Compressor(hidden_size_,
                                   head_dim_,
                                   rope_head_dim_,
                                   compress_ratio_,
                                   cached_state_num,
                                   eps_,
                                   /*rotate=*/false,
                                   rotary_emb_,  // TODO: pass proper rotary_emb
                                   options));

    // Indexer only for compress_ratio == 4
    if (compress_ratio_ == 4) {
      indexer_ = register_module("indexer",
                                 IndexerV2(hidden_size_,
                                           args.index_n_heads(),
                                           args.index_head_dim(),
                                           rope_head_dim_,
                                           args.index_topk(),
                                           q_lora_rank_,
                                           window_size_,
                                           compress_ratio_,
                                           cached_state_num,
                                           eps_,
                                           rotary_emb_,
                                           parallel_args,
                                           options));
    }
  }
}

void DeepSeekV4AttentionImpl::write_kv_to_cache(
    const torch::Tensor& kv,
    torch::Tensor& k_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& seq_lens,
    bool is_prefill) {
  int64_t batch_size = seq_lens.size(0);

  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t query_start = q_cu_seq_lens[i].item<int64_t>();
    int64_t query_end = q_cu_seq_lens[i + 1].item<int64_t>();
    int64_t seqlen = seq_lens[i].item<int64_t>();

    auto kv_seq = kv.slice(0, query_start, query_end);

    if (is_prefill) {
      if (seqlen <= window_size_) {
        // Sequence fits in window: direct copy
        auto slot_mapping =
            block_table.index({i, torch::indexing::Slice(0, seqlen)})
                .to(torch::kInt32);
        kernel::ReshapePagedCacheParams params;
        params.key = kv_seq;
        params.k_cache = k_cache;
        params.slot_mapping = slot_mapping;
        kernel::reshape_paged_cache(params);
      } else {
        // Sequence exceeds window: use ring buffer layout
        int64_t cutoff = seqlen % window_size_;
        auto kv_tail = kv_seq.slice(0, -window_size_);
        auto slot_mapping =
            block_table.index({i, torch::indexing::Slice(0, window_size_)})
                .to(torch::kInt32);

        if (cutoff > 0) {
          auto kv_parts = kv_tail.split({window_size_ - cutoff, cutoff}, 0);
          // Write head part
          kernel::ReshapePagedCacheParams params1;
          params1.key = kv_parts[0];
          params1.k_cache = k_cache;
          params1.slot_mapping = slot_mapping.slice(0, cutoff, window_size_);
          kernel::reshape_paged_cache(params1);
          // Write tail part
          kernel::ReshapePagedCacheParams params2;
          params2.key = kv_parts[1];
          params2.k_cache = k_cache;
          params2.slot_mapping = slot_mapping.slice(0, 0, cutoff);
          kernel::reshape_paged_cache(params2);
        } else {
          kernel::ReshapePagedCacheParams params;
          params.key = kv_tail;
          params.k_cache = k_cache;
          params.slot_mapping = slot_mapping;
          kernel::reshape_paged_cache(params);
        }
      }
    } else {
      // Decode: write single token to ring buffer position
      int64_t slot_idx = (seqlen - 1) % window_size_;
      auto slot_mapping =
          block_table.index({i, torch::indexing::Slice(slot_idx, slot_idx + 1)})
              .to(torch::kInt32);
      kernel::ReshapePagedCacheParams params;
      params.key = kv_seq;
      params.k_cache = k_cache;
      params.slot_mapping = slot_mapping;
      kernel::reshape_paged_cache(params);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
DeepSeekV4AttentionImpl::convert_topk_to_block_tables(
    const std::vector<torch::Tensor>& topk_idxs_list,
    int64_t window_size,
    int64_t max_model_len,
    int64_t compress_ratio,
    const std::optional<torch::Tensor>& offsets,
    const torch::Tensor& block_tables,
    bool is_prefill) {
  // Compute total tokens
  int64_t total_token = 0;
  for (const auto& seq : topk_idxs_list) {
    total_token += seq.size(0);
  }

  // Block table width
  int64_t block_table_width =
      window_size + (compress_ratio > 0 ? max_model_len / compress_ratio : 0);

  auto device = topk_idxs_list[0].device();
  auto new_block_tables =
      torch::full({total_token, block_table_width},
                  -1,
                  torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto new_context_lens =
      torch::zeros({total_token},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  int64_t token_offset = 0;
  for (size_t bz = 0; bz < topk_idxs_list.size(); ++bz) {
    const auto& seq_topk = topk_idxs_list[bz];
    int64_t seq_len = seq_topk.size(0);
    for (int64_t i = 0; i < seq_len; ++i) {
      auto token_idxs = seq_topk[i];  // [K]
      auto valid_mask = token_idxs != -1;
      auto valid_idxs = token_idxs.index({valid_mask});
      int64_t num_valid = valid_idxs.numel();

      CHECK(num_valid > 0 && num_valid <= block_table_width)
          << "Invalid number of valid indices: " << num_valid;

      int64_t pos = token_offset + i;
      new_context_lens[pos] = static_cast<int32_t>(num_valid);

      if (is_prefill) {
        int64_t batch_offset =
            offsets.has_value() ? offsets.value()[bz].item<int64_t>() : 0;
        new_block_tables.index_put_(
            {pos, torch::indexing::Slice(0, num_valid)},
            valid_idxs.slice(0, 0, num_valid).to(torch::kInt32) +
                static_cast<int32_t>(batch_offset));
      } else {
        // For decode mode, valid_idxs is already int32 from topk_idxs_list
        // Convert to int64 for indexing block_tables, then back to int32
        auto valid_idxs_int64 =
            valid_idxs.slice(0, 0, num_valid).to(torch::kInt64);
        auto block_indices =
            block_tables.index({static_cast<int64_t>(bz), valid_idxs_int64});
        // Ensure block_indices is int32 before assignment
        if (block_indices.dtype() != torch::kInt32) {
          block_indices = block_indices.to(torch::kInt32);
        }
        new_block_tables.index_put_({pos, torch::indexing::Slice(0, num_valid)},
                                    block_indices);
      }
    }
    token_offset += seq_len;
  }

  return {new_block_tables, new_context_lens};
}

torch::Tensor DeepSeekV4AttentionImpl::forward_sparse_attn(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    torch::Tensor& k_cache,
    torch::Tensor& indexer_cache,
    const AttentionMetadata& attn_metadata,
    const std::vector<int64_t>& batch_to_kv_state) {
  int64_t num_tokens = hidden_states.size(0);
  auto device = hidden_states.device();
  bool is_prefill = attn_metadata.is_prefill;

  const auto& q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  const auto& seq_lens = attn_metadata.kv_seq_lens;
  const auto& block_tables = attn_metadata.block_table;
  int64_t batch_size = seq_lens.size(0);

  // Query projection with LoRA
  torch::Tensor q, qr;
  if (q_lora_rank_ > 0) {
    auto q_a = wq_a_->forward(hidden_states);
    auto [q_normed, _] = q_norm_->forward(q_a);
    qr = q_normed;
    q = wq_b_->forward(q_normed);
    q = q.view({-1, num_local_heads_, head_dim_});

    // Apply RMSNorm to q (per-head normalization)
    auto q_squared = q.square();
    auto q_mean = q_squared.mean(-1, /*keepdim=*/true);
    q = q * torch::rsqrt(q_mean + eps_);
  }

  // KV projection
  auto kv = wkv_->forward(hidden_states);
  auto [kv_normed, __] = kv_norm_->forward(kv);
  kv = kv_normed.unsqueeze(-2);  // [num_tokens, 1, head_dim]

  // Apply RoPE to q_pe and kv_pe
  torch::Tensor q_pe = q.slice(-1, head_dim_ - rope_head_dim_, head_dim_);
  torch::Tensor kv_pe = kv.slice(-1, head_dim_ - rope_head_dim_, head_dim_);
  rotary_emb_->forward(
      q_pe, positions, q_cu_seq_lens, attn_metadata.max_query_len, is_prefill);
  rotary_emb_->forward(
      kv_pe, positions, q_cu_seq_lens, attn_metadata.max_query_len, is_prefill);

  // Write KV to paged cache
  write_kv_to_cache(
      kv, k_cache, block_tables, q_cu_seq_lens, seq_lens, is_prefill);

  // Get window top-k indices
  auto window_topk_idxs =
      get_window_topk_idxs(window_size_, q_cu_seq_lens, seq_lens, device);

  // Compute offsets for compress indices
  auto query_lens = q_cu_seq_lens.slice(0, 1) - q_cu_seq_lens.slice(0, 0, -1);
  torch::Tensor offsets;
  if (is_prefill) {
    offsets = query_lens;
  } else {
    offsets = torch::full_like(query_lens, window_size_);
  }

  // Get compressed KV indices
  std::vector<torch::Tensor> compress_topk_idxs;
  std::vector<torch::Tensor> compress_kvs;
  std::vector<int64_t> compress_lens;
  torch::Tensor context_lens;  // Will be set in prefill mode with compression

  if (compress_ratio_ > 0) {
    if (indexer_) {
      // Use indexer for dynamic top-k selection
      compress_topk_idxs = indexer_->forward(hidden_states,
                                             qr,
                                             positions,
                                             offsets,
                                             attn_metadata,
                                             batch_to_kv_state,
                                             indexer_cache,
                                             freqs_cis_);

    } else {
      // Use static compress indices
      compress_topk_idxs = get_compress_topk_idxs(
          compress_ratio_, q_cu_seq_lens, seq_lens, offsets, device);
    }
    // Run compressor to get compressed KV

    auto [kvs, lens] = compressor_->forward(hidden_states,
                                            positions,
                                            block_tables,
                                            q_cu_seq_lens,
                                            seq_lens,
                                            batch_to_kv_state,
                                            k_cache,
                                            window_size_,
                                            freqs_cis_);
    compress_kvs = kvs;
    compress_lens = lens;

    // Merge KV for prefill
    if (is_prefill) {
      // Update context lens: context_lens = seq_lens + compress_lens
      auto compress_lens_tensor =
          torch::tensor(compress_lens, seq_lens.options());
      context_lens = seq_lens + compress_lens_tensor;

      // Merge original KV with compressed KV
      std::vector<torch::Tensor> merge_kvs;
      for (int64_t i = 0; i < batch_size; ++i) {
        int64_t query_start = q_cu_seq_lens[i].item<int64_t>();
        int64_t query_end = q_cu_seq_lens[i + 1].item<int64_t>();
        auto cur_kv = kv.slice(0, query_start, query_end);
        if (compress_lens[i] > 0) {
          auto kv_merge = torch::cat({cur_kv, compress_kvs[i]}, 0);
          merge_kvs.push_back(kv_merge);
        } else {
          merge_kvs.push_back(cur_kv);
        }
      }
      kv = torch::cat(merge_kvs, 0);
    }
  }

  // Combine window and compress indices
  std::vector<torch::Tensor> topk_idxs_list;
  for (int64_t i = 0; i < batch_size; ++i) {
    torch::Tensor topk_idxs;
    if (compress_ratio_ > 1) {
      topk_idxs = torch::cat({window_topk_idxs[i], compress_topk_idxs[i]}, -1);
    } else {
      topk_idxs = window_topk_idxs[i];
    }
    topk_idxs_list.push_back(topk_idxs.to(torch::kInt32));
  }

  // Compute batch_offset for convert_topk_to_block_tables
  // - Prefill + compress_ratio > 0: cu_context_lens = [0, cumsum(context_lens)]
  // - Prefill + compress_ratio == 0: query_start_loc (q_cu_seq_lens)
  // - Decode: None
  std::optional<torch::Tensor> batch_offset = std::nullopt;
  if (is_prefill) {
    if (compress_ratio_ > 0 && context_lens.defined()) {
      auto cu_context_lens =
          torch::cat({torch::zeros({1}, context_lens.options()),
                      torch::cumsum(context_lens, 0)});
      batch_offset = cu_context_lens;
    } else {
      batch_offset = q_cu_seq_lens;
    }
  }

  // Convert top-k indices to block tables
  auto [new_block_tables, new_context_lens] =
      convert_topk_to_block_tables(topk_idxs_list,
                                   window_size_,
                                   max_model_len_,
                                   compress_ratio_,
                                   batch_offset,
                                   block_tables,
                                   is_prefill);

  // Prepare for sparse attention
  auto attn_output = torch::zeros_like(q);
  int64_t total_token = q.size(0);

  auto q_reshaped = q.view({total_token, 1, num_local_heads_, head_dim_});
  auto out_reshaped =
      attn_output.view({total_token, 1, num_local_heads_, head_dim_});

  torch::Tensor kv_cache_reshaped;
  if (is_prefill) {
    kv_cache_reshaped = kv.unsqueeze(1);  // [N, 1, 1, head_dim]
  } else {
    kv_cache_reshaped = k_cache;
  }

  // Run sparse attention using batch_decode kernel
  int64_t max_context_len =
      window_size_ +
      (compress_ratio_ > 0 ? attn_metadata.max_seq_len / compress_ratio_ : 0);

  // Pre-allocate output_lse tensor for attention sink computation
  // Shape: [batch, num_heads, seq_q] -> [total_token, num_local_heads_, 1]
  auto output_lse = torch::empty(
      {total_token, num_local_heads_, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  std::optional<torch::Tensor> output_lse_opt = output_lse;

  xllm::kernel::mlu::batch_decode(q_reshaped,
                                  kv_cache_reshaped,
                                  out_reshaped,
                                  new_block_tables,
                                  new_context_lens,
                                  /*v_cache=*/std::nullopt,
                                  output_lse_opt,
                                  /*q_quant_scale=*/std::nullopt,
                                  /*k_cache_quant_scale=*/std::nullopt,
                                  /*v_cache_quant_scale=*/std::nullopt,
                                  /*out_quant_scale=*/std::nullopt,
                                  /*alibi_slope=*/std::nullopt,
                                  /*mask=*/std::nullopt,
                                  /*compute_dtype=*/"float",
                                  max_context_len,
                                  /*window_size_left=*/-1,
                                  /*window_size_right=*/-1,
                                  softmax_scale_,
                                  /*return_lse=*/true,
                                  /*kv_cache_quant_bit_size=*/-1);

  // Apply attention sink
  // output_lse: [N, H, 1] -> transpose(1,2) -> [N, 1, H] -> unsqueeze(-1) ->
  // [N, 1, H, 1]
  auto lse_transposed =
      output_lse.transpose(1, 2).contiguous().unsqueeze(-1);  // [N, 1, H, 1]
  auto sinks = attn_sink_.reshape({1, 1, -1, 1});             // [1, 1, H, 1]
  auto multiplier =
      1.0 / (torch::exp(sinks - lse_transposed) + 1.0);  // [N, 1, H, 1]
  // out_reshaped is [N, 1, H, D], multiply with [N, 1, H, 1]
  attn_output = out_reshaped * multiplier;  // [N, 1, H, D]
  // Reshape output
  attn_output = attn_output.reshape({-1, num_local_heads_, head_dim_});

  // Apply inverse RoPE to output
  torch::Tensor attn_output_pe =
      attn_output.slice(-1, head_dim_ - rope_head_dim_, head_dim_);
  output_rotary_emb_->forward(attn_output_pe,
                              positions,
                              q_cu_seq_lens,
                              attn_metadata.max_query_len,
                              is_prefill);

  // Convert to output dtype
  attn_output = attn_output.to(torch::kBFloat16);

  // Output projection
  torch::Tensor output;
  if (use_parallel_o_proj_) {
    attn_output = attn_output.reshape({num_tokens, o_local_groups_, -1});
    auto wo_a_weight =
        wo_a_col_->weight().view({o_local_groups_, o_lora_rank_, -1});
    auto o = torch::einsum("ngd,grd->ngr", {attn_output, wo_a_weight});
    output = wo_b_row_->forward(o.flatten(-2));
    output = parallel_state::reduce(output, tp_group_);
  } else {
    attn_output = attn_output.flatten(-2).contiguous();
    attn_output = parallel_state::gather(attn_output, tp_group_, -1);
    attn_output = attn_output.reshape({-1, num_heads_, head_dim_}).contiguous();
    auto wo_a_weight = wo_a_rep_->weight().view({o_groups_, o_lora_rank_, -1});
    attn_output = attn_output.reshape({num_tokens, o_groups_, -1});
    auto o = torch::einsum("ngd,grd->ngr", {attn_output, wo_a_weight});
    output = wo_b_rep_->forward(o.flatten(-2));
  }

  return output;
}

torch::Tensor DeepSeekV4AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const std::vector<int64_t>& batch_to_kv_state) {
  auto k_cache = kv_cache.get_k_cache();
  auto indexer_cache = kv_cache.get_index_cache();
  return forward_sparse_attn(positions,
                             hidden_states,
                             k_cache,
                             indexer_cache,
                             attn_metadata,
                             batch_to_kv_state);
}

void DeepSeekV4AttentionImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = tp_rank_;
  const int64_t world_size = tp_size_;

  // Load query projection
  if (wq_a_) {
    wq_a_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  }

  // Load KV projection
  wkv_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_norm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));

  // Load output projection
  if (use_parallel_o_proj_) {
    wo_a_col_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
    wo_b_row_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));
  } else {
    wo_a_rep_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
    wo_b_rep_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));
  }

  // Load attention sink
  LOAD_SHARDED_WEIGHT(attn_sink, 0);

  // Load compressor
  if (compressor_) {
    compressor_->load_state_dict(
        state_dict.get_dict_with_prefix("compressor."));
  }

  // Load indexer
  if (indexer_) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }
}

}  // namespace layer
}  // namespace xllm
