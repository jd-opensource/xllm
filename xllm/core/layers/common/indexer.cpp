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
      softmax_scale_(std::pow(head_dim_, -0.5) * std::pow(n_heads_, -0.5)) {
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

  k_norm_ = register_module(
      "k_norm",
      torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_})
                               .eps(1e-6)
                               .elementwise_affine(true)));
  // set the device of k_norm_ to the same as the options
  k_norm_->to(options.device());

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

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    const std::optional<torch::Tensor>& mask) {
  // Forward pass through wq_b
  auto q = wq_b_->forward(qr);
  q = q.reshape({q.size(0), n_heads_, head_dim_});

  // Forward pass through wk and normalize
  auto k = wk_->forward(x);
  k = k_norm_->forward(k);

  // Split q and k into positional encoding and non-positional encoding parts
  // (like Python)
  auto q_split =
      torch::split(q, {rope_head_dim_, head_dim_ - rope_head_dim_}, -1);
  auto q_pe = q_split[0].contiguous();
  auto q_nope = q_split[1].contiguous();

  auto k_split =
      torch::split(k, {rope_head_dim_, head_dim_ - rope_head_dim_}, -1);
  auto k_pe = k_split[0].contiguous();
  auto k_nope = k_split[1].contiguous();

  // Apply rotary embedding to positional parts only (like Python)
  auto k_pe_unsqueezed = k_pe.unsqueeze(1);
  rotary_emb_->forward(q_pe,
                       k_pe_unsqueezed,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);
  k_pe = k_pe_unsqueezed.squeeze(1);

  // Reconstruct q and k
  q_pe = q_pe.reshape({q_pe.size(0), n_heads_, rope_head_dim_});
  q = torch::cat({q_pe, q_nope}, -1);
  k = torch::cat({k_pe, k_nope}, -1);

  // Apply rotation activation
  q = rotate_activation(q, hadamard_matrix_);
  k = rotate_activation(k, hadamard_matrix_);

  // Forward pass through weights projection
  auto weights = weights_proj_->forward(x);

  // kv_cache part
  int64_t num_tokens = x.size(0);

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
  // Unified parameter setup for both prefill and decode modes
  torch::Tensor k_cache_tensor;
  std::optional<torch::Tensor> cu_seq_q_lens, k_block_table;
  torch::Tensor new_block_tables;
  int64_t batch_size = attn_metadata.kv_seq_lens.size(0);
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor cu_seq_k_lens = attn_metadata.q_cu_seq_lens;

  if (is_prefill) {
    // Prefill mode parameters
    cu_seq_q_lens = attn_metadata.q_cu_seq_lens;
    k_block_table = std::nullopt;
    k_cache_tensor = k;

    // Prefill output tensors
    new_block_tables = torch::empty({num_tokens, index_topk_},
                                    torch::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(block_table.device()));
  } else {
    // Decode mode parameters
    cu_seq_q_lens = std::nullopt;
    k_block_table = block_table;
    k_cache_tensor = k_cache;
    auto seq_len = num_tokens / batch_size;

    // Reshape tensors for decode mode
    q = q.view({batch_size, seq_len, n_heads_, head_dim_});
    weights = weights.view({batch_size, seq_len, n_heads_});

    // Decode output tensors
    new_block_tables = torch::empty({batch_size, seq_len, index_topk_},
                                    torch::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(block_table.device()));
  }
  auto new_context_lens = torch::empty(
      {num_tokens},
      torch::TensorOptions().dtype(torch::kInt32).device(block_table.device()));

  // Call masked indexer select paged kv
  kernel::MaskedIndexerSelectPagedKVParams params;
  params.is_prefill = is_prefill;
  params.query = q;
  params.cu_seq_q_lens = cu_seq_q_lens.value_or(torch::Tensor());
  params.cu_seq_k_lens = cu_seq_k_lens;
  params.q_scale = torch::Tensor();  // empty tensor as q_scale
  params.weights = weights;
  params.softmax_scale = softmax_scale_;
  params.k_cache = k_cache_tensor;
  params.k_context_lens = attn_metadata.kv_seq_lens;
  params.k_cache_block_table = k_block_table.value_or(torch::Tensor());
  params.k_scale_cache = torch::Tensor();  // empty tensor as k_scale_cache
  params.index_topk = index_topk_;
  params.kv_cache_block_table = block_table;
  params.kv_cache_block_size = 1;  // only support 1 for now
  params.new_block_table = new_block_tables;
  params.new_context_lens = new_context_lens;
  params.quant_block_size = 128;  // only support 128 for now

  xllm::kernel::masked_indexer_select_paged_kv(params);

  if (!is_prefill) {
    new_block_tables = new_block_tables.view({-1, new_block_tables.size(-1)});
  }

  return {new_block_tables, new_context_lens};
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
}

}  // namespace layer
}  // namespace xllm
