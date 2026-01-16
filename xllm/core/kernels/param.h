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
#include <tvm/ffi/container/array.h>

#include <optional>
#include <string>
#include <vector>

namespace ffi = tvm::ffi;

namespace xllm::kernel {

// Note: add default values for optional parameters in the struct definition

// Rotary embedding parameters
struct RotaryParams {
  // Query tensor. First dimension is total_seq_len (T).
  // Will be reshaped to [T, -1] and concatenated with k before applying rotary
  // embedding. Head size must be between 2 and 256.
  torch::Tensor q;
  // Key tensor. First dimension must match q.size(0) (total_seq_len).
  // Will be reshaped to [T, -1] and concatenated with q before applying rotary
  // embedding.
  torch::Tensor k;
  // Sin cache tensor for rotary embedding. Shape:
  // - [rope_seqlen, rope_dim] if dynamic_ntk=false
  // - [batch_size, rope_seqlen, rope_dim] if dynamic_ntk=true
  // rope_dim must be between 2 and head_size, and must be even.
  // rope_dim is extracted as sin.size(-1) and used to reshape qk tensor.
  torch::Tensor sin;
  // Cos cache tensor for rotary embedding. Same shape as sin.
  // The rope_seqlen-stride must equal to sin's rope_seqlen-stride.
  torch::Tensor cos;
  // Precomputed cos_sin tensor. Not used in current MLU implementation
  // (rope.cpp).
  torch::Tensor cos_sin;
  // Optional position IDs tensor. Type must be int32.
  // Shape: [total_seqlen] if discrete=true, or [batch_size] if discrete=false.
  // If discrete=true, position_ids must be provided.
  std::optional<torch::Tensor> position_ids;
  // Cumulative query lengths tensor. Type must be int32, must be contiguous.
  // Required in pack mode (when q/k are 3D). Size should be [batch_size + 1].
  // Note: In current MLU implementation, this is always passed to underlying
  // API.
  std::optional<torch::Tensor> cu_query_lens;
  // Whether to use interleaved rotary embedding pattern.
  bool interleaved;
  // Whether to use discrete position mode. If true, position_ids must be
  // provided and have shape [total_seqlen]. If false, position_ids can be None
  // or have shape [batch_size].
  bool discrete;
  // Whether to use dynamic NTK (Neural Tangent Kernel) scaling.
  // If true, sin and cos caches must have batch dimension.
  // Note: Current MLU implementation hardcodes this to false when calling
  // underlying API, so dynamic_ntk=true may not be fully supported.
  bool dynamic_ntk = false;
  // Maximum query length. In pad mode (4D input), must equal to input.size(1).
  // Must be less than or equal to rope_seqlen if not using discrete
  // position_ids.
  int64_t max_query_len;
};

// Activation parameters
struct ActivationParams {
  // Input tensor. Must be contiguous, dimension >= 2.
  // Last dimension is in_channel, which must be > 0.
  // If is_gated=true, in_channel must be even.
  torch::Tensor input;
  // Output tensor. Must be contiguous, dimension >= 2.
  // Must have same attributes (device, dtype) as input.
  // Only supports stride in dim(-2), stride(-1) must be 1.
  // Shape: [total_tokens, inner_size] where inner_size = in_channel/2 if
  // is_gated else in_channel.
  torch::Tensor output;
  // Optional bias tensor, only used for MoE activation.
  // If provided, cusum_token_count must also be provided.
  // Shape: [expert_size, in_channel]. Must be contiguous.
  std::optional<torch::Tensor> bias;
  // Optional cumulative token count tensor. Type should be int32.
  // Required when bias is provided. Must be contiguous.
  // Size: [num_expert + 1], where num_expert = size(0) - 1.
  std::optional<torch::Tensor> cusum_token_count;
  // Activation mode string. Must be one of: "silu", "gelu", "quick_gelu",
  // "swish".
  // - "silu": SiLU activation (Swish-1)
  // - "gelu": GELU activation
  // - "quick_gelu": Quick GELU with coefficient 1.702
  // - "swish": Swish activation
  std::string act_mode;
  // Whether to use gated activation. If true, input's last dimension
  // (in_channel) must be even, and output's inner_size will be in_channel/2.
  bool is_gated;
  // Starting expert ID for MoE activation. Used when processing multiple
  // experts.
  int64_t start_expert_id = 0;
  // Expert size for MoE activation. Used when bias is provided.
  // Bias tensor shape must be [expert_size, in_channel].
  int64_t expert_size = 0;
};

// Reshape paged cache parameters
struct ReshapePagedCacheParams {
  // Key tensor from context. Shape: [num_tokens, num_heads, head_dim].
  // Last two dimensions must be contiguous: stride(-1)==1,
  // stride(-2)==head_dim. Must have same device and dtype as k_cache and
  // v_cache.
  torch::Tensor key;
  // Optional value tensor from context. Shape: [num_tokens, num_heads,
  // head_dim]. If provided, v_cache must also be provided (and vice versa).
  // Last two dimensions must be contiguous: stride(-1)==1,
  // stride(-2)==head_dim. Must have same device and dtype as other tensors.
  std::optional<torch::Tensor> value;
  // Key cache tensor in paged format. Shape: [num_blocks, num_heads,
  // block_size, head_dim]. Must be contiguous. Must have same device and dtype
  // as key and value.
  torch::Tensor k_cache;
  // Optional value cache tensor in paged format. Shape: [num_blocks, num_heads,
  // block_size, head_dim]. If provided, value must also be provided (and vice
  // versa). Must be contiguous. Must have same device and dtype as other
  // tensors.
  std::optional<torch::Tensor> v_cache;
  // Slot mapping tensor. Shape: [num_tokens]. Type must be int32.
  // Maps each token to its corresponding slot in the cache. Must be contiguous.
  // Must have same device as key.
  torch::Tensor slot_mapping;
  // Direction flag: false = CONTEXT2CACHE (copy from context to cache),
  // true = CACHE2CONTEXT (copy from cache to context).
  bool direction = false;
};

// Attention parameters
// Note: This struct is used by both batch_prefill (flash_attention) and
// batch_decode (single_query_cached_kv_attn). Parameters are grouped by usage.
struct AttentionParams {
  // ========== Common parameters (used by both prefill and decode) ==========
  // Query tensor. Shape depends on mode:
  // - Prefill: 3D [total_tokens, num_heads, head_dim] (packed) or
  //            4D [batch, seq_len, num_heads, head_dim] (padded)
  // - Decode: 4D [batch, seq_q, num_heads, head_dim]
  //   Last two dims must be contiguous: stride(-1)==1, stride(-2)==head_dim
  torch::Tensor query;
  // Output tensor. Shape: [batch, seq_q, num_heads, head_dim] (decode) or
  // [total_tokens, num_heads, head_dim] (prefill packed) or
  // [batch, seq_len, num_heads, head_dim] (prefill padded).
  // Last two dims must be contiguous: stride(-1)==1, stride(-2)==head_dim
  torch::Tensor output;
  // Optional log-sum-exp output tensor. Shape: [batch, num_heads, seq_q].
  // Used when return_lse=true. Must be contiguous.
  std::optional<torch::Tensor> output_lse;
  // Optional ALiBi (Attention with Linear Biases) slope tensor.
  // - Prefill: shape [batch, num_heads] or [num_heads]
  // - Decode: shape [batch, num_heads] or [num_heads]
  std::optional<torch::Tensor> alibi_slope;
  // Optional query quantization scale tensor.
  // - Prefill: 1D (fp8 per-tensor) or 3D (sage per-block)
  // - Decode: 1D [1] (per-tensor) or 3D [batch, seq_q, num_heads] (per-token)
  // If provided, k_quant_scale (prefill) or k_cache_quant_scale (decode) must
  // also be provided.
  std::optional<torch::Tensor> q_quant_scale;
  // Optional output quantization scale. Currently not supported (must be None).
  // Reserved for future use.
  std::optional<torch::Tensor> out_quant_scale;
  // Optional block table for paged KV cache. Type: int32 or int64. Must be
  // contiguous.
  // - Prefill: 2D [batch, max_num_blocks_per_seq] (if used)
  // - Decode: 2D [batch, max_num_blocks_per_seq] or
  //           3D [batch, num_kv_heads, max_num_blocks_per_seq]
  std::optional<torch::Tensor> block_table;
  // Compute data type string. Must be "float", "half", or "bfloat16".
  std::string compute_dtype;
  // Maximum sequence length. Used for workspace allocation and attention
  // computation.
  // - Prefill: max_seq_len_kv
  // - Decode: max_context_len
  int64_t max_seq_len;
  // Left window size for sliding window attention. Must be >= 0.
  int64_t window_size_left;
  // Right window size for sliding window attention. Default: -1.
  // In decode mode, only supports window_size_right < 0 currently.
  int64_t window_size_right = -1;
  // Softmax scaling factor. Applied to Q@K^T before softmax.
  float scale;
  // Whether to return log-sum-exp values in output_lse.
  bool return_lse = false;
  // ========== Torch NPU related parameters ==========
  torch::Tensor seq_lens;
  torch::Tensor attn_mask;

  // ========== FlashInfer related parameters ==========
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor float_workspace_buffer;
  torch::Tensor int_workspace_buffer;
  torch::Tensor page_locked_int_workspace_buffer;
  std::string uri;
  ffi::Array<int64_t> plan_info;

  bool enable_cuda_graph = false;
  // Whether to use tensor core for decode attention computation. Default: true.
  bool use_tensor_core = true;

  // ========== Prefill-specific parameters ==========
  // Key tensor. Shape: [num_tokens, num_kv_heads, head_dim_qk] (packed) or
  // [batch, seq_len, num_kv_heads, head_dim_qk] (padded) or
  // [num_blocks, num_kv_heads, block_size, head_dim_qk] (paged with
  // block_table). If block_table provided, must be 4D. Must have same dim as
  // value.
  torch::Tensor key;
  // Value tensor. Shape: [num_tokens, num_kv_heads, head_dim_vo] (packed) or
  // [batch, seq_len, num_kv_heads, head_dim_vo] (padded) or
  // [num_blocks, num_kv_heads, block_size, head_dim_vo] (paged with
  // block_table). If block_table provided, must be 4D. Must have same dim as
  // key.
  torch::Tensor value;
  // Optional cumulative query sequence lengths. Type: int32.
  // Shape: [batch + 1]. Required in packed mode (query is 3D). Must be
  // contiguous.
  std::optional<torch::Tensor> q_cu_seq_lens;
  // Optional cumulative KV sequence lengths. Type: int32.
  // Shape: [batch + 1]. Required in packed mode or when block_table has
  // max_num_blocks_per_seq > 1. Must be contiguous.
  std::optional<torch::Tensor> kv_cu_seq_lens;
  // Query sequence lengths.Shape: [batch]. Type: int32. Must be contiguous.
  // Represents the current query length for each sequence in the batch.
  torch::Tensor q_seq_lens;
  // KV sequence lengths tensor. Shape: [batch]. Type: int32. Must be
  // contiguous. Represents the current context length for each sequence in the
  // batch.
  torch::Tensor kv_seq_lens;
  // Optional attention bias tensor. Used for custom attention patterns.
  std::optional<torch::Tensor> attn_bias;
  // Optional key quantization scale tensor.
  // - 1D [1]: fp8 per-tensor quantization (requires q_quant_scale to be 1D)
  // - 3D: sage per-block quantization (not supported with block_table)
  // Must have same dim as q_quant_scale.
  std::optional<torch::Tensor> k_quant_scale;
  // Optional value quantization scale tensor.
  // - 1D [1]: fp8 per-tensor quantization (requires q_quant_scale to be 1D)
  // - 3D [batch, num_kv_heads, head_dim_vo]: sage per-channel quantization
  std::optional<torch::Tensor> v_quant_scale;
  // Maximum query length. Used for workspace allocation in prefill.
  int64_t max_query_len;
  // Whether to apply causal mask. Default: true.
  bool is_causal = true;

  // ========== Decode-specific parameters ==========
  // Key cache tensor in paged format. Shape: [num_blocks, num_kv_heads,
  // block_size, head_dim]. Must be contiguous. If kv_cache_quant_bit_size=4,
  // shape is [num_blocks, num_kv_heads, block_size, head_dim/2].
  torch::Tensor k_cache;
  // Value cache tensor in paged format. Shape: [num_blocks, num_kv_heads,
  // block_size, head_dim]. Must be contiguous. If kv_cache_quant_bit_size=4,
  // shape is [num_blocks, num_kv_heads, v_cache_len, head_dim] where
  // v_cache_len = PAD_UP_DIV(block_size, 2).
  std::optional<torch::Tensor> v_cache;
  // Optional key cache quantization scale tensor. Must be contiguous.
  // - 2D [num_kv_heads, head_dim]: per-channel quantization
  // - 3D [num_blocks, num_kv_heads, block_size]: per-token quantization
  // - 4D [num_blocks, num_kv_heads, block_size, 1]: per-token quantization
  // Required when q_quant_scale is provided.
  std::optional<torch::Tensor> k_cache_quant_scale;
  // Optional value cache quantization scale tensor. Must be contiguous.
  // Must have same dim as k_cache_quant_scale if provided.
  // - 2D [num_kv_heads, head_dim]: per-channel quantization
  // - 3D [num_blocks, num_kv_heads, block_size]: per-token quantization
  // - 4D [num_blocks, num_kv_heads, block_size, 1]: per-token quantization
  std::optional<torch::Tensor> v_cache_quant_scale;
  // Optional attention mask tensor.
  // When provided, uses NTT (non-token-to-token) mask mode instead of causal
  // mask.
  std::optional<torch::Tensor> mask;
  // KV cache quantization bit size. Default: -1 (no quantization).
  // Supported values: -1 (no quant), 4 (int4), 8 (int8).
  // If 4, k_cache and v_cache shapes are adjusted for int4 packing.
  int64_t kv_cache_quant_bit_size = -1;
};

// Fused layer norm parameters
struct FusedLayerNormParams {
  // Input tensor. Dimension must be >= 2. Last dimension is hidden_size.
  // Last dimension must be contiguous: stride(-1) == 1.
  // Must have same device and dtype as residual, weight, beta, bias,
  // residual_out, normed_out.
  torch::Tensor input;
  // Output tensor. Must have same shape as input.
  // If inplace (input.data_ptr() == output.data_ptr()), strides must also be
  // the same. Must have same device as input, smooth_quant_scale, quant_scale.
  torch::Tensor output;
  // Optional residual tensor. Must have same shape as input.
  // If provided, must have same device and dtype as input.
  std::optional<torch::Tensor> residual;
  // Weight tensor (gamma). Shape: [hidden_size]. Must be contiguous.
  // Required for both layernorm and rmsnorm modes.
  // Must have same device and dtype as input.
  torch::Tensor weight;
  // Optional beta tensor. Shape: [hidden_size]. Must be contiguous.
  // Required for layernorm mode, not used in rmsnorm mode.
  // If provided, must have same dtype as weight.
  std::optional<torch::Tensor> beta;
  // Optional bias tensor. Shape: [hidden_size]. Must be contiguous.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> bias;
  // Optional quantization scale tensor. Type must be float.
  // Shape: [hidden_size] (1D) or [head, headdim] (2D).
  // - 1D: per-channel quantization, input will be flattened to 2D
  // - 2D: only supported for rmsnorm mode, input must be dim >= 3,
  //       shape must be [head, headdim], residual and bias not supported
  // If dynamic_quant=true, this must be provided.
  std::optional<torch::Tensor> quant_scale;
  // Optional residual output tensor. Used when store_output_before_norm=true.
  // Not supported when both bias and residual are not provided.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> residual_out;
  // Optional smooth quantization scale tensor. Type must be float.
  // Used when dynamic_quant=true. Will be flattened to 1D.
  // Must have same device as input.
  std::optional<torch::Tensor> smooth_quant_scale;
  // Optional normalized output tensor. Used when store_output_after_norm=true.
  // Only supported when dynamic_quant=true.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> normed_out;
  // Normalization mode. Must be "layernorm" or "rmsnorm".
  // - "layernorm": requires both weight (gamma) and beta
  // - "rmsnorm": only requires weight (gamma), beta is not used
  std::string mode;
  // Epsilon value for numerical stability in normalization computation.
  double eps;
  // Whether to store output before normalization to residual_out.
  // Not supported when both bias and residual are not provided.
  bool store_output_before_norm = false;
  // Whether to store output after normalization to normed_out.
  // Only supported when dynamic_quant=true.
  bool store_output_after_norm = false;
  // Whether to use dynamic quantization. If true, quant_scale must be provided.
  // When true, uses per-token quantization scheme; otherwise uses per-channel
  // if quant_scale provided.
  bool dynamic_quant = false;
};

// Matmul parameters
struct MatmulParams {
  // Left input tensor A. Must be 2D or 3D. Must have same dimension as b.
  // Must have same dtype as b.
  // For 2D: shape [M, K], output will be [M, N] where N = b.size(-1)
  // For 3D: shape [batch, M, K], output will be [batch, M, N]
  // If input dtype is int8 or fp8, c must be provided to determine output
  // dtype.
  torch::Tensor a;
  // Right input tensor B. Must be 2D or 3D. Must have same dimension as a.
  // Must have same dtype as a.
  // For 2D: shape [K, N], output will be [M, N] where M = a.size(-2)
  // For 3D: shape [batch, K, N], output will be [batch, M, N]
  torch::Tensor b;
  // Optional bias tensor. Will be added to the matrix multiplication result.
  std::optional<torch::Tensor> bias;
  // Optional output tensor C. Can be used to specify output dtype and
  // accumulate result. If input dtype is int8 or fp8, c or dtype must be
  // provided to determine output dtype. If provided, result will be: output =
  // alpha * (a @ b) + beta * c
  std::optional<torch::Tensor> c;
  // Scaling factor for matrix multiplication result. Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double alpha = 1.0;
  // Scaling factor for tensor c (if provided). Default: 0.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double beta = 0.0;
};

struct GroupGemmParams {
  // Input activation tensor.
  // Shape: 2D [M, K] if trans_a==false; [K, M] if trans_a==true.
  // Must be contiguous. Dtype: float16, bfloat16, or float32.
  // Must have same dtype and device as b, output.
  torch::Tensor a;
  // Weight tensor.
  // If trans_b is true, shape is (num_experts, N, K) or (N, K);
  // if trans_b is false, shape is (num_experts, K, N) or (K, N).
  // Must be contiguous. Dtype and device must match a, output.
  torch::Tensor b;
  // Per-expert token count tensor.
  // Shape: 1D [num_experts]. Type must be int32.
  // Controls number of tokens processed per group/expert.
  torch::Tensor token_count;
  // Output tensor.
  // Shape: [num_experts, N] or [num_experts, N, K]. num_experts =
  // token_count.size(0). Must be contiguous. Dtype and device must match a.
  torch::Tensor output;
  // Optional scale tensor for a (input activation), used in quantized mode.
  // Shape depends on quantization granularity.
  std::optional<torch::Tensor> a_scale;
  // Optional scale tensor for b (weight), used in quantized mode.
  // Shape depends on quantization granularity.
  std::optional<torch::Tensor> b_scale;
  // Optional quantization config flag list.
  // Used to control per-expert weight quantization mode.
  std::optional<torch::List<int64_t>> quant_flag;
  // Maximum workspace dimension (e.g., maximum tokens per expert allowed).
  // Used for configuring inner kernel workspace.
  int64_t max_dim;
  // Whether to transpose a:
  // false: [M, K] (default); true: [K, M].
  bool trans_a;
  // Whether to transpose b:
  // false: [K, N] (default); true: [N, K].
  bool trans_b;
  // Quantization bit-width for input a.
  // Set -1 to disable quantization.
  int64_t a_quant_bit;
};

struct MoeActiveTopkParams {
  // Input tensor.
  // Shape: [*, num_mask, num_expert] (e.g., [batch, num_mask, num_expert]).
  // Dtype: float32, float16, bfloat16.
  // Must be contiguous.
  torch::Tensor input;
  // Number of top-k experts to select per token.
  // Constraint: 0 < topk <= num_expert.
  int64_t topk;
  // Number of expert groups for group-limited top-k selection.
  // If > 1, mask must be None, and num_expert % num_expert_group == 0.
  int64_t num_expert_group;
  // Maximum selected experts per group.
  // Constraint: 0 < topk_group <= num_expert_group.
  int64_t topk_group;
  // Whether to renormalize expert weights after top-k selection.
  bool normalize;
  // Optional mask tensor.
  // Shape: [1, ..., 1, num_mask, num_expert] (leading dims must be 1).
  // Dtype must match input.
  // Must be contiguous.
  std::optional<torch::Tensor> mask;
  // Normalization logic after top-k selection.
  // For softmax: "topk_logit" or "softmax_logit".
  // For sigmoid: "topk_logit" or "sigmoid_logit".
  std::string normed_by;
  // Scoring function for expert selection.
  // Supported: "softmax", "sigmoid".
  std::string scoring_func;
  // Route scaling factor applied to routing scores.
  double route_scale;
  // Optional expert score correction bias.
  // Shape: [num_expert].
  // Dtype: float32, float16, or bfloat16.
  // Must be contiguous.
  std::optional<torch::Tensor> e_score_correction_bias;
};

struct MoeGenIdxParams {
  // The input tensor stores the expert id of each token.
  // Shape: [num_tokens, topk].
  // Dtype: int32.
  torch::Tensor expert_id;
  // Expert number.
  // Must be >= 0.
  int64_t expert_num;
};

struct MoeExpandInputParams {
  // Input tensor to be expanded.
  // Shape: [token_num, hidden_size].
  // Dtype: int8, float, half, or bfloat16.
  torch::Tensor input;
  // Index tensor for gather operation.
  // Shape: [expand_token_num].
  // Dtype: int32.
  torch::Tensor gather_index;
  // Optional prefix sum of token count per expert.
  // Shape: [num_experts + 1].
  // Dtype: int32.
  // If provided, adjusts gather range for each expert.
  std::optional<torch::Tensor> cusum_token_count;
  // Starting expert id to process.
  // Must be >= 0.
  int64_t start_expert_id;
  // Number of experts to process in this call.
  // Must be >= 0.
  int64_t expert_size;
};

struct MoeCombineResultParams {
  // Expert output tensor to be combined.
  // Shape: [num_tokens * topk, hidden_size].
  // - Must be contiguous.
  // - Dtype: float32, float16, or bfloat16.
  // - This is the concatenated output from all experts, not yet reordered back
  // to the original sequence order.
  torch::Tensor input;
  // Router/gating weights tensor. Used for weighted combination of expert
  // outputs. Shape: [num_tokens, topk].
  // - Must be contiguous at last dimension.
  // - Dtype: float32.
  // - Constraint: reduce_weight.numel() == input.size(0).
  torch::Tensor reduce_weight;
  // Gather index tensor that maps combined output to original token positions.
  // Shape: [num_tokens * topk].
  // - Must be contiguous.
  // - Dtype: int32.
  // - Corresponds to permutation/scatter indices for reordering expert outputs.
  torch::Tensor gather_ids;
  // Optional residual connection input.
  // Shape: [num_tokens, hidden_size].
  // - Must have same shape and dtype as output if provided.
  // - Must be contiguous if provided.
  // - Default: std::nullopt (no residual).
  std::optional<torch::Tensor> residual;
  // Optional cumulative token count for expert assignment.
  // Shape: [num_experts + 1] or deduced by expert_size.
  // - Must be contiguous if provided.
  // - Dtype: int32.
  // - Used to infer num_expert or assist calculation in some kernels.
  std::optional<torch::Tensor> cusum_token_count;
  // Starting expert ID
  // - Must be >= 0.
  // - Used to mark the offset of current experts being processed (for
  // sharding).
  int64_t start_expert_id = 0;
  // Number of experts processed in this step.
  // - If cusum_token_count not given, num_expert is set to this value.
  // - If cusum_token_count given, deduced num_expert must satisfy:
  //   num_expert >= start_expert_id + expert_size
  int64_t expert_size = 0;
  // Optional bias tensor.
  // WARNING: Bias addition is NOT supported in current implementation.
  // Always keep as std::nullopt unless bias support is added in the future.
  std::optional<torch::Tensor> bias;
};

struct MoeAll2AllGenSendLayoutParams {
  // Expert token count tensor.
  // Shape: [expert_num].
  // Dtype: int32.
  // Each element represents the number of tokens assigned to each expert.
  torch::Tensor token_count;
  // Number of ranks (processes) participating in All2All.
  // Must be >= 0.
  int64_t nrank;
};

struct MoeAll2AllGenGatherIndexParams {
  // The table that indicates the relationship of token for each Expert Parallel
  // part. Shape: [rank_num, expert_num], where rank_num is the number of
  // devices in Expert Parallel, and expert_num is the number of experts handled
  // by each device. Dtype: int32.
  torch::Tensor token_num;
  // The max token count for each rank (used for padding).
  // Dtype: int32. Must be >= 0.
  int64_t pad_num;
  // Whether to return the cusum_token_count tensor.
  // If true, cusum_token_count will be returned.
  bool return_cusum_token_count = false;
};

struct MoeAll2AllCreateParams {
  // Byte size of a single token for dispatch All-to-All operation.
  // Each token to be dispatched requires this many bytes.
  int64_t dispatch_token_byte;
  // Byte size of a single token for combine All-to-All operation.
  // Each token to be combined requires this many bytes.
  int64_t combine_token_byte;
  // Maximum number of experts participating in the All-to-All operation.
  // (Sets the upper bound for how many experts can be involved.
  int64_t max_expert_num;
  // Maximum number of tokens to be processed.
  // Upper bound on the total batch size in tokens for the operation.
  int64_t max_token_num;
  // Rank ID of the current process in the distributed group, within [0,
  // nrank-1]. Identifies this process within the world group.
  int64_t rank;
  // Total number of processes in the distributed group.
  // Used for collective communication context and split assignment.
  int64_t nrank;
  // The current compute device to be used、
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

struct MoeAll2AllInitParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // CPU tensor containing aggregated exchange information from all nrank
  // processes.
  torch::Tensor all_exchange_info;
  // The current compute device to be used
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

struct MoeAll2AllDispatchParams {
  // Communication backend handle for All-to-All operation.
  // Obtained from moe_all2all_create.
  int64_t handle;
  // Byte size of a single token.
  int64_t token_byte;
  // Number of tokens to be processed in the current operation.
  int64_t token_num;
  // Offset and token count for each rank.
  // The token_count is generated by moe_gen_idx.
  // Shape: [nrank, 2]. Type: int32.
  torch::Tensor send_layout;
  // Number of tokens to send to each expert.
  // Shape: [max_expert_num]. Type: int32.
  torch::Tensor send_token_num;
  // Offset and token count from peer ranks.
  // Shape: [nrank, 2]. Type: int32.
  torch::Tensor recv_layout;
  // Expected number of tokens to receive from each expert.
  // Shape: [max_expert_num]. Type: int32.
  torch::Tensor recv_token_num;
  // Optional tensor containing tokens to dispatch.
  // If not provided, defaults to dispatch_send created by moe_all2all_create.
  std::optional<torch::Tensor> send_token;
  // Optional buffer for receiving tokens.
  // If not provided, defaults to dispatch_recv created by moe_all2all_create.
  std::optional<torch::Tensor> recv_token;
};

struct MoeAll2AllCombineParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // Byte size of a single token.
  int64_t token_byte;
  // The number of tokens to receive.
  int64_t token_num;
  // The offset and token count for each rank, output from
  // Shape: [nrank, 2],
  // Type: int32.
  torch::Tensor send_src_layout;
  // The expected receive pattern from peer ranks.
  // Shape: [nrank, 2],
  // Type: int32.
  torch::Tensor send_dst_layout;
  // Optional tensor containing the tokens to dispatch. If not provided,
  // defaults to combine_send created by moe_all2all_create.
  std::optional<torch::Tensor> send_token;
  // Optional buffer for receiving tokens. If not provided,
  // defaults to combine_recv created by moe_all2all_create.
  std::optional<torch::Tensor> recv_token;
};

struct MoeAll2AllDestroyParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // The current compute device to be used
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

// Per token smooth quantize parameters
// Note: Current MLU implementation uses "dynamic_per_token" quantization mode.
struct ScaledQuantizeParams {
  // Input tensor to quantize. Dimension must be >= 2.
  // Must be continuous between 0 and -2 dimensions (can be flattened to 2D).
  // If gather_index or token_count has value, x must be 2D.
  // Must have same device as other tensors.
  torch::Tensor x;
  // Smooth quantization scale tensor (corresponds to x_scale in underlying
  // API). Shape constraints depend on quantization mode and other parameters.
  // - If token_count has value: shape [token_count.size(0),
  // x.size(-1)/(1+is_gated)]
  // - If is_gated: smooth.size(-1) * 2 == x.size(-1)
  // - Otherwise: smooth.size(-1) == x.size(-1)
  // Must be contiguous if provided. Must have same device as x.
  torch::Tensor smooth;
  // Zero point tensor. Must be None (not supported in current implementation).
  std::optional<torch::Tensor> zero;
  // Optional token count tensor when quantizing MoE group gemm inputs.
  // If provided, x must be 2D and smooth.size(0) must equal
  // token_count.size(0). Must be contiguous if provided. Must have same device
  // as x.
  std::optional<torch::Tensor> token_count;
  // Optional gather index tensor when quantizing MoE group gemm inputs. Shape:
  // [output_tokens]. If provided, x must be 2D. Output shape will be adjusted:
  // output_shape[0] = gather_index.size(0). If gather_index_start_position is
  // provided, gather_index must also be provided. Must be contiguous if
  // provided. Must have same device as x.
  std::optional<torch::Tensor> gather_index;
  // Optional gather index start position tensor when quantizing MoE group gemm
  // inputs. Only used if gather_index is provided. Must be contiguous if
  // provided. Must have same device as x.
  std::optional<torch::Tensor> gather_index_start_position;
  // Optional output tensor when quantizing MoE group gemm inputs.
  // Type must be int8 (kChar), float8_e4m3fn, or float8_e5m2.
  // Dimension must be >= 2. Must be continuous between 0 and -2 dimensions.
  // Shape constraints:
  // - If !gather_index && !is_gated: output.sizes() == x.sizes()
  // - If is_gated: output.size(-1) * 2 == x.size(-1)
  // - If gather_index: output_shape[0] = gather_index.size(0)
  // If not provided, will be allocated automatically with quant_type.
  // Must have same device as x.
  std::optional<torch::Tensor> output;
  // Optional output scale tensor.
  // Used in dynamic_per_token quantization mode.
  // Shape: x.sizes()[0:-1] (same as x except last dimension removed).
  // If gather_index provided: shape[0] = gather_index.size(0).
  // Must be flattenable to 1D with numel == output_flat.size(0).
  // If not provided, will be allocated automatically with float32 dtype.
  // Must have same device as x.
  std::optional<torch::Tensor> output_scale;
  // Activation mode. Must be one of: "none", "gelu", "silu", "swish".
  // Default: "none". If "none", is_gated will be set to false automatically.
  // If "silu", active_coef will be set to 1.0 automatically.
  std::string act_mode = "none";
  // Activation coefficient. Default: 1.0.
  // If act_mode == "silu", this will be set to 1.0 automatically.
  double active_coef = 1.0;
  // Whether to use gated activation. Default: false.
  // If act_mode == "none", this will be set to false automatically.
  // If true, output's last dimension will be x.size(-1) / 2.
  bool is_gated = false;
  // Quantization output data type. Default: torch::kChar (int8).
  // Supported: torch::kChar (int8), torch::kFloat8_e4m3fn, torch::kFloat8_e5m2.
  torch::ScalarType quant_type = torch::kChar;
};

// Scaled matmul parameters
// Note: Current MLU implementation only supports:
// - smooth_quant algorithm
// - w8a8 quantization (quant_bit_size=8, a_quant_bit_size=8)
// - trans_a=false, trans_b=true (hardcoded)
struct ScaledMatmulParams {
  // Input tensor A. Shape: [M, K]. Must be contiguous.
  // Output shape will be [M, N] where N = b.size(0).
  // Must have same device as other tensors.
  torch::Tensor a;
  // Weight tensor B. Shape: [K, N]. Will be transposed (trans_b=true).
  // Must be contiguous. Must have same device as other tensors.
  torch::Tensor b;
  // Optional scale tensor for A. Shape: 1D or 2D. Must be contiguous or have
  // stride (1, m).
  // - 1D: per-token quantization layout
  // - 2D: group-wise quantization layout
  // Note: In current MLU implementation (scaled_matmul.cpp), a_scale is
  // required.
  std::optional<torch::Tensor> a_scale;
  // Scale tensor for B. Shape: 1D or 2D. Must be contiguous or have stride (1,
  // n). Determines quantization layout:
  // - 1D: per-channel quantization
  // - 2D: per-block (if b_scale.size(0) < b.size(0)) or group-wise quantization
  // Must be contiguous. Must have same device as other tensors.
  torch::Tensor b_scale;
  // Output data type. Must be torch::kFloat16 (half) or torch::kBFloat16.
  torch::ScalarType output_dtype;
  // Optional bias tensor. Will be added to the matrix multiplication result.
  // Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> bias;
  // Optional tensor C for accumulation. Result: alpha * (a @ b) + beta * c.
  // Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> c;
  // Activation mode. Default: "none". Supported: "none", "silu", "gelu".
  // If "silu", act_coef will be set to 1.0 automatically.
  std::string act_mode = "none";
  // Quantization bit size for B (weight). Default: 8.
  // Current implementation only supports 8 (w8a8 quantization).
  // Supported values: 4, 8.
  int64_t quant_bit_size = 8;
  // Scaling factor for matrix multiplication result. Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double alpha = 1.0;
  // Scaling factor for tensor c (if provided). Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double beta = 1.0;
  // Whether to use high precision activation computation. Default: false
  // If true, uses high precision; otherwise uses fast computation.
  bool use_hp_active = false;
  // Quantization bit size for A (activation). Default: -1.
  // Current implementation only supports 8 (w8a8 quantization).
  // Supported values: -1 (no quantization), 4, 8.
  int64_t a_quant_bit_size = -1;
  // Optional calibration tensor for A. Used for flat_quant and svd_quant
  // algorithms. Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> a_calib;
  // Optional calibration tensor for B. Used for flat_quant and svd_quant
  // algorithms. Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> b_calib;
  // Optional output tensor. Shape: [M, N] where M = a.size(0), N = b.size(0).
  // If not provided, will be allocated automatically with output_dtype.
  // Must have same device as other tensors.
  std::optional<torch::Tensor> output;
};

// Top-K and Top-P sampling parameters
struct TopKPParams {
  // Input logits tensor. Shape: [batch_size, vocab_size]. Type must be float32.
  // Must be contiguous. Will be converted to float32 if needed.
  // If both top_k and top_p are not defined, logits will be returned directly.
  torch::Tensor logits;
  // Temperature tensor for scaling logits. Shape: [batch_size].
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor temperatures;
  // Optional top-k values tensor. Type will be converted to int32.
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor top_k;
  // Optional top-p (nucleus sampling) values tensor.
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor top_p;
};

// Random sample parameters
struct RandomSampleParams {
  // Input tensor of probabilities for sampling.
  // Must be 2-dimensional: [batch_size, vocab_size]
  torch::Tensor logits;
};

// Masked indexer select paged KV cache parameters
struct MaskedIndexerSelectPagedKVParams {
  // Whether this is prefill phase (true) or decode phase (false).
  // Affects query shape and whether cu_seq_q_lens is used.
  bool is_prefill;
  // Query tensor. Must have same dtype as k_cache (bfloat16, half, or int8).
  // - Prefill mode: 3D [total_seq_q, head_num, head_size], head_num must be 64
  // - Decode mode: 4D [batch_num, len_q, head_num, head_size], head_num must be
  // 64 Does not need to be contiguous
  torch::Tensor query;
  // Cumulative sequence lengths for queries. Type: int32. Must be contiguous.
  // Required in prefill mode, not used in decode mode.
  torch::Tensor cu_seq_q_lens;
  // Cumulative sequence lengths for keys.
  torch::Tensor cu_seq_k_lens;
  // Query quantization scale tensor. Must be contiguous.
  // - Required (numel > 0) when query dtype is int8 or fp8
  // - Must be empty (numel == 0) when query dtype is bfloat16 or half
  torch::Tensor q_scale;
  // Attention weights tensor. Dtype must be bfloat16 or float32. Must be
  // contiguous.
  torch::Tensor weights;
  // Softmax scaling factor for attention computation.
  double softmax_scale;
  // Key cache tensor in paged format. Shape: [num_blocks, 1, block_size,
  // head_dim]. Dim(1) must be 1. Must be contiguous. Must have same dtype as
  // query.
  torch::Tensor k_cache;
  // Key context lengths tensor. Shape: [batch_num]. Type: int32. Must be
  // contiguous.
  torch::Tensor k_context_lens;
  // Key cache block table. Shape: [batch_num, k_cache_max_blkn]. Type: int32.
  // Must be contiguous.
  torch::Tensor k_cache_block_table;
  // Key cache quantization scale tensor. Must be contiguous.
  // - Required (numel > 0) when k_cache dtype is int8 or fp8
  // - Must be empty (numel == 0) when k_cache dtype is bfloat16 or half
  torch::Tensor k_scale_cache;
  // Number of top-k indices to select. Must be >= 0.
  int64_t index_topk;
  // KV cache block table. Shape: [batch_num, kv_cache_max_blkn]. Type: int32.
  // Must be contiguous.
  torch::Tensor kv_cache_block_table;
  // KV cache block size. Must be 1.
  int64_t kv_cache_block_size;
  // New block table output tensor. Must be contiguous.
  // - Prefill mode: 2D [total_seq_q, kv_cache_max_blkn]
  // - Decode mode: 3D [batch_num, seq_q, kv_cache_max_blkn]
  torch::Tensor new_block_table;
  // New context lengths output tensor. Shape: [batch_num] (prefill) or
  // [batch_num] (decode). Type: int32. Must be contiguous.
  torch::Tensor new_context_lens;
  // Quantization block size. Must equal to head_size (query.size(-1)).
  int64_t quant_block_size;
};

struct GatherSplitParams {
  // Input tensor. Shape: (token_num, input_size).
  // Dtype: int8, float32, float16, or bfloat16.
  torch::Tensor input;
  // Gather index tensor. Shape: (token_num).
  // Dtype: int32.
  // Used to select valid tokens from the input tensor.
  torch::Tensor gather_index;
  // Number of valid tokens tensor. Shape: (1).
  // Dtype: int32.
  // Its first element is the actual valid token count: valid_token_num =
  // valid_token_num[0].item().
  torch::Tensor valid_token_num;
  // Output tensor for the "head" split. Shape: (token_num, size_0).
  // Dtype: same as input.
  // Holds the gathered and split tokens for the first size_0 elements of each
  // token.
  torch::Tensor output_head;
  // Optional output tensor for the "tail" split. Shape: (token_num, input_size
  // - size_0). Dtype: same as input. If provided, holds the gathered and split
  // tokens for the remaining elements after size_0.
  // Pass empty tensor to skip the tail split.
  torch::Tensor output_tail;
};

}  // namespace xllm::kernel
