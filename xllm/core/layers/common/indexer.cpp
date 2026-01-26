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
#include "indexer.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>

#include "kernels/ops_api.h"

namespace {

// Returns true if n is a power of two (greater than 0 and only one bit set)
inline bool is_power_of_two(int64_t n) { return n > 0 && ((n & (n - 1)) == 0); }

// Generates an n×n Hadamard matrix (Sylvester type, elements ±1).
// If normalize = true, returns orthogonal Hadamard: H / sqrt(n).
torch::Tensor create_hadamard_matrix(int64_t n,
                                     torch::Dtype dtype = torch::kFloat32,
                                     torch::Device device = torch::kCPU,
                                     bool normalize = false) {
  CHECK(is_power_of_two(n)) << "hadamard_matrix: n must be a power of two.";

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  // Initial Hadamard matrix H_1 = [1]
  torch::Tensor H = torch::ones({1, 1}, options);

  // Recursively build Hadamard: H_{2m} = [[H_m,  H_m], [H_m, -H_m]]
  for (int64_t m = 1; m < n; m <<= 1) {
    // Concatenate along column (dim=1) for top and bottom blocks
    auto top = torch::cat({H, H}, /*dim=*/1);
    auto bottom = torch::cat({H, -H}, /*dim=*/1);
    // Concatenate along row (dim=0) to form next Hadamard matrix
    H = torch::cat({top, bottom}, /*dim=*/0);
  }

  if (normalize) {
    H = H / std::sqrt(static_cast<double>(n));
  }
  return H;
}

// Performs a Hadamard-like linear transform with optional zero-padding and
// scaling.
//
// Args:
//   x: Tensor of shape (..., dim)
//   h_matrix: Tensor of shape (dim_padded, dim_padded). Treated as
//   [out_features, in_features], matching Python F.linear weight convention.
//   scale: Optional multiplicative scaling factor (default = 1.0). By default,
//   no scaling is applied (matches Python version).
//
// Returns:
//   Tensor of same shape as x, after transformation.
torch::Tensor hadamard_transform_ref(const torch::Tensor& x,
                                     const torch::Tensor& h_matrix) {
  // Save original shape and input dimension
  const auto x_shape = x.sizes();
  const int64_t dim = x.size(-1);
  // Flatten x to 2D of shape [-1, dim]
  torch::Tensor x2d = x.reshape({-1, dim});
  // Compute next power of two for padding
  const double log_dim = std::ceil(std::log2(static_cast<double>(dim)));
  // 2 ** log_dim
  const int64_t dim_padded =
      static_cast<int64_t>(1ull << static_cast<uint64_t>(log_dim));
  // Pad the last dimension with zeros on the right to reach dim_padded if
  // necessary
  if (dim != dim_padded) {
    // Padding order: [pad_left_dim, pad_right_dim, ...], applied from last
    // dimension backwards
    x2d = torch::nn::functional::pad(
        x2d,
        torch::nn::functional::PadFuncOptions({0, dim_padded - dim})
            .mode(torch::kConstant)
            .value(0));
  }
  // Linear transformation: F.linear(input, weight) with no bias
  // weight should have shape [out_features, in_features]; so out = x2d @
  // h_matrix.T
  torch::Tensor out = torch::nn::functional::linear(x2d, h_matrix);

  // Truncate result to original dim (last dimension)
  using torch::indexing::Slice;
  out = out.index({Slice(), Slice(0, dim)});
  // Restore original shape
  return out.reshape(x_shape);
}
}  // namespace

namespace xllm {
namespace layer {

IndexerImpl::IndexerImpl(int64_t dim,
                         int64_t index_n_heads,
                         int64_t index_head_dim,
                         int64_t qk_rope_head_dim,
                         int64_t index_topk,
                         int64_t q_lora_rank,
                         bool enable_fused_qk,
                         DeepseekScalingRotaryEmbedding& rotary_emb,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options)
    : dim_(dim),
      n_heads_(index_n_heads),
      head_dim_(index_head_dim),
      rope_head_dim_(qk_rope_head_dim),
      index_topk_(index_topk),
      q_lora_rank_(q_lora_rank),
      rotary_emb_(rotary_emb),
      softmax_scale_(std::pow(head_dim_, -0.5) * std::pow(n_heads_, -0.5)),
      enable_fused_qk_(enable_fused_qk) {
  // Note: The current Indexer implementation does not yet support quantization
  // or parallelization strategies. These features are planned for future
  // updates. For now, the entire indexer computation runs independently on each
  // MLU on any parallel strategy.

  // Register modules
  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(q_lora_rank,
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  wk_ = register_module("wk",
                        ReplicatedLinear(dim,
                                         head_dim_,
                                         /*bias=*/false,
                                         quant_args,
                                         options));

  weights_proj_ = register_module("weights_proj",
                                  ReplicatedLinear(dim,
                                                   n_heads_,
                                                   /*bias=*/false,
                                                   quant_args,
                                                   options));

  // the default eps is defined as 1e-6 in indexer implementation of
  // DeepSeek-V3.2.
  double default_eps = 1e-6;
  k_norm_ = register_module(
      "k_norm",
      RMSNorm(head_dim_, default_eps, options.dtype(torch::kFloat32)));
  k_norm_->set_layernorm_mode();

  // Create hadamard matrix
  int64_t head_dim_padded = std::pow(2, std::ceil(std::log2(head_dim_)));
  // Construct the Hadamard matrix on CPU with float32, then cast to target
  // dtype and device set normalize=true is equivalent to scale=hidden_size **
  // -0.5
  hadamard_matrix_ = create_hadamard_matrix(
      head_dim_padded, torch::kFloat32, torch::kCPU, true);
  hadamard_matrix_ =
      hadamard_matrix_.to(options.device(), options.dtype().toScalarType());
}

torch::Tensor IndexerImpl::rotate_activation(
    const torch::Tensor& input,
    const torch::Tensor& hadamard_matrix) {
  // Ensure the input is bfloat16 as per interface contract
  CHECK(input.dtype() == torch::kBFloat16)
      << "rotate_activation: input must be bfloat16";
  int64_t hidden_size = input.size(-1);
  return hadamard_transform_ref(input, hadamard_matrix);
}

IndexerRuntimeContext IndexerImpl::prepare_runtime_context(
    const torch::Tensor& k_current_dense,
    torch::Tensor& k_cache_paged,
    torch::Tensor& q,
    torch::Tensor& weights,
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    int64_t num_tokens) {
  IndexerRuntimeContext ctx;
  auto device = attn_metadata.block_table.device();

  // Allocate context_lens buffer
  ctx.new_context_lens = torch::empty(
      {num_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  if (is_prefill) {
    // Prefill: flatten Q and weights
    ctx.q = q;
    ctx.weights = weights;
    ctx.cu_seq_q_lens = attn_metadata.q_cu_seq_lens;
    ctx.k_block_table = std::nullopt;

    ctx.new_block_tables = torch::empty(
        {num_tokens, index_topk_},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (attn_metadata.is_chunked_prefill) {
      // NOTE: the kv_cu_seq_lens should already include the history tokens
      ctx.cu_seq_k_lens = attn_metadata.kv_cu_seq_lens;

      // Allocate contiguous memory for gathered k
      int64_t total_k_len = ctx.cu_seq_k_lens[-1].item<int64_t>();
      ctx._storage_k_full = torch::empty(
          {total_k_len, head_dim_},
          torch::TensorOptions().dtype(k_cache_paged.dtype()).device(device));

      // Calculate sequence lengths by diff of offsets
      auto seq_lens = torch::diff(ctx.cu_seq_k_lens);
      int64_t max_context_len = seq_lens.max().item<int64_t>();

      ctx.k_context_lens = seq_lens;

      // Gather k from cache
      xllm::kernel::ReshapeFromCacheParams gather_params;
      gather_params.key = ctx._storage_k_full.unsqueeze(1);
      gather_params.value = std::nullopt;
      gather_params.key_cache = k_cache_paged;
      gather_params.value_cache = std::nullopt;
      gather_params.context_lengths = seq_lens;
      gather_params.max_context_len = max_context_len;
      gather_params.block_tables = attn_metadata.block_table;
      gather_params.context_seq_offset = std::nullopt;
      gather_params.cache_seq_offset = std::nullopt;

      xllm::kernel::reshape_from_cache(gather_params);
      ctx.k_cache_tensor = ctx._storage_k_full;
    } else {
      // Standard prefill: k is dense
      ctx.cu_seq_k_lens = attn_metadata.q_cu_seq_lens;
      ctx.k_context_lens = attn_metadata.kv_seq_lens;
      ctx.k_cache_tensor = k_current_dense;
    }
  } else {
    // Decode mode
    int64_t batch_size = attn_metadata.kv_seq_lens.size(0);
    auto seq_len = num_tokens / batch_size;

    // Reshape q and weights for decode
    ctx.q = q.view({batch_size, seq_len, n_heads_, head_dim_});
    ctx.weights = weights.view({batch_size, seq_len, n_heads_});

    ctx.new_block_tables = torch::empty(
        {batch_size, seq_len, index_topk_},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    ctx.cu_seq_q_lens = std::nullopt;
    ctx.cu_seq_k_lens = attn_metadata.q_cu_seq_lens;
    ctx.k_block_table = attn_metadata.block_table;
    ctx.k_cache_tensor = k_cache_paged;
    ctx.k_context_lens = attn_metadata.kv_seq_lens;
  }

  return ctx;
}

torch::Tensor IndexerImpl::preprocess_indexer_q(
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  // Forward pass through wq_b
  auto q = wq_b_->forward(qr);
  q = q.view({q.size(0), n_heads_, head_dim_});
  auto q_pe = q.slice(-1, 0, rope_head_dim_);
  rotary_emb_->forward(q_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);

  // Apply rotation activation
  q = rotate_activation(q, hadamard_matrix_);
  return q;
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::preprocess_indexer_k(
    const torch::Tensor& x,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata) {
  // Forward pass through wk and normalize
  auto k = wk_->forward(x);
  auto k_dtype = k.dtype();
  // follow the implementation of DeepSeek-V3.2,
  // the k_norm is applied on the float32 tensor.
  auto k_fp32 = k.to(torch::kFloat32);
  k = std::get<0>(k_norm_->forward(k_fp32)).to(k_dtype);

  // Apply rotary embedding to positional parts only (like Python)
  auto k_pe = k.slice(-1, 0, rope_head_dim_).unsqueeze(1);
  rotary_emb_->forward(k_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);
  k = rotate_activation(k, hadamard_matrix_);

  // Reshape paged cache
  auto k_unsqueezed = k.unsqueeze(1);
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = k_unsqueezed;
  reshape_paged_cache_params.value = std::nullopt;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = std::nullopt;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  reshape_paged_cache_params.direction = false;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  k = k_unsqueezed.squeeze(1);

  // Forward pass through weights projection
  auto weights = weights_proj_->forward(x);

  return {k, weights};
}

torch::Tensor IndexerImpl::preprocess_indexer_q_fused(
    const torch::Tensor& qr,
    const torch::Tensor& positions) {
  // fuses the query projection(Matmul), Rotary Position Embedding (RoPE), and
  // an optional Hadamard transformation(Matmul) into a single high-performance
  // kernel
  auto output = torch::empty({qr.size(0), n_heads_, head_dim_}, qr.options());
  auto w_q = wq_b_->weight().view({n_heads_, head_dim_, -1});
  kernel::FusedIndexerQParams q_params;
  q_params.input_q = qr;
  q_params.output = output;
  q_params.output_scale = std::nullopt;
  q_params.w_q = w_q;
  q_params.w_q_scale = std::nullopt;
  q_params.hadamard_matrix = hadamard_matrix_;
  q_params.sin = rotary_emb_->get_sin_cache();
  q_params.cos = rotary_emb_->get_cos_cache();
  q_params.position_id = positions;
  q_params.quant_mode = "none";
  kernel::fused_indexer_q(q_params);
  return output;
}

torch::Tensor IndexerImpl::preprocess_indexer_k_fused(
    const torch::Tensor& x,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata) {
  // Perform wk(x), layernorm, rope, wproj(x) and quant to paged k_cache
  auto wproj_weight = weights_proj_->weight();
  auto head_weights =
      torch::empty({x.size(0), wproj_weight.size(0)}, x.options());
  kernel::FusedIndexerKParams k_params;
  k_params.x = x;
  k_params.wk = wk_->weight();
  k_params.wproj = wproj_weight;
  k_params.sin_table = rotary_emb_->get_sin_cache();
  k_params.cos_table = rotary_emb_->get_cos_cache();
  k_params.position_id = positions;
  k_params.slot_mapping = attn_metadata.slot_mapping;
  k_params.head_weights = head_weights;
  k_params.k_cache = k_cache;
  k_params.k_cache_scale = std::nullopt;
  k_params.hadamard_matrix = hadamard_matrix_;
  kernel::fused_indexer_k(k_params);
  return head_weights;
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    const std::optional<torch::Tensor>& mask) {
  torch::Tensor q, k, weights;
  if (!is_prefill && enable_fused_qk_) {
    q = preprocess_indexer_q_fused(qr, positions);
    weights = preprocess_indexer_k_fused(x, positions, k_cache, attn_metadata);
  } else {
    q = preprocess_indexer_q(qr, positions, attn_metadata);
    std::tie(k, weights) =
        preprocess_indexer_k(x, positions, k_cache, attn_metadata);
  }
  // Unified parameter setup for both prefill and decode modes
  IndexerRuntimeContext ctx = prepare_runtime_context(
      k, k_cache, q, weights, attn_metadata, is_prefill, x.size(0));

  // Call masked indexer select paged kv
  kernel::MaskedIndexerSelectPagedKVParams params;
  params.query = ctx.q;
  params.k_cache = ctx.k_cache_tensor;
  params.weights = ctx.weights;
  params.kv_cache_block_table = attn_metadata.block_table;
  params.cu_seq_q_lens = ctx.cu_seq_q_lens;
  params.cu_seq_k_lens = ctx.cu_seq_k_lens;
  params.k_context_lens = ctx.k_context_lens;
  params.k_cache_block_table = ctx.k_block_table;
  params.is_prefill = is_prefill;
  params.softmax_scale = softmax_scale_;
  params.q_scale = std::nullopt;        // empty tensor as q_scale
  params.k_scale_cache = std::nullopt;  // empty tensor as k_scale_cache
  params.index_topk = index_topk_;
  params.kv_cache_block_size = FLAGS_block_size;
  params.sparse_block_table = ctx.new_block_tables;
  params.sparse_context_lens = ctx.new_context_lens;

  xllm::kernel::masked_indexer_select_paged_kv(params);

  if (!is_prefill) {
    ctx.new_block_tables =
        ctx.new_block_tables.view({-1, ctx.new_block_tables.size(-1)});
  }

  return {ctx.new_block_tables, ctx.new_context_lens};
}

// load the weight from the checkpoint
void IndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  // Load weights for each linear layer
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  wk_->load_state_dict(state_dict.get_dict_with_prefix("wk."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
}

}  // namespace layer
}  // namespace xllm
