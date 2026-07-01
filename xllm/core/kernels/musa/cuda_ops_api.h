/* Copyright 2025-2026 The xLLM Authors.

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

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <glog/logging.h>

#include <optional>
#include <tuple>
#include <vector>

#include "kernels/musa/utils.h"

namespace xllm::kernel::cuda {

// TODO: add head_size parameter
void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      // int64_t head_size,
                      bool is_neox);

// act_mode only support silu, gelu, gelu_tanh
void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

// out[i] *= sigmoid(gate[i]) in-place. No allocations (graph-capture safe).
void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate);

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache);

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype);

#if !defined(USE_DCU)
void batch_prefill(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& mask = std::nullopt);

// Wrapper function for batch_prefill that conditionally uses AttentionRunner
// for piecewise CUDA Graph capture
void batch_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse);

void batch_prefill_non_causal(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& mask = std::nullopt);

void batch_chunked_prefill(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    bool causal = true);

// `paged_kv_*_host` are optional pre-staged CPU mirrors of the corresponding
// device tensors. When defined, the Mate FFI bridge consumes them directly
// and skips an internal .to(kCPU) sync. AttentionMetadataBuilder fills these
// once per forward step (see attention_metadata.h); callers that do not have
// pre-staged hosts can leave them undefined and the legacy lazy D2H runs.
void batch_decode(const std::string& uri,
                  ffi::Array<int64_t> plan_info,
                  torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool use_tensor_core,
                  std::optional<torch::Tensor> qo_indptr = std::nullopt,
                  const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
                  const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
                  const torch::Tensor& paged_kv_last_page_len_host =
                      torch::Tensor());
// FA3 unified attention decode (single-pass, warp-specialized). Calls the
// JIT-built `fmha_fwd_<hash>.so` from mate's cached_ops. See fa3_fwd.cpp.
//
// Inputs:
//   query:              [total_q, num_qo_heads, head_dim_qk]   bf16 device
//   k_cache, v_cache:   [n_pages, page_size, num_kv_heads, head_dim]  bf16
//   cu_seqlens_q:       [batch + 1]   int32 device
//   seqused_k:          [batch]       int32 device (per-seq kv length)
//   page_table:         [batch, max_pages_per_seq] int32 device (-1 padded;
//                       built by batch_input_builder from allocated KV blocks)
//   scheduler_metadata: int32 metadata tensor produced by
//                       update_fa3_decode_plan_info() at layer 0
//   max_seqlen_q:       1 for plain decode; >1 for MTP spec-verify
//   window_left:        sliding-window left edge (-1 disables)
//   window_right:       sliding-window right edge (0 = causal mask only)
//   sm_scale:           softmax scale = 1 / sqrt(head_dim)
// Outputs (preallocated):
//   output:    [total_q, num_qo_heads, head_dim_vo] bf16
//   output_lse:[num_qo_heads, total_q]              fp32
void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse);

// Precompute scheduler_metadata for fa3_decode. Call once per shape (per
// PlanInfo) at layer 0; reuse the returned tensor across all decode layers.
torch::Tensor fa3_decode_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_k);

#endif  // !defined(USE_DCU)
void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon);

// Gemma-style RMS norm: same as `rms_norm` but applies `(weight + 1.0)` as
// the per-element scale, fusing the `+1.0` into the kernel so the caller
// does NOT need to materialize `(1.0 + weight)` on the host. Required for
// MUSA graph capture safety -- the pure-torch fallback at
// `cuda_fallback::gemma_rms_norm` allocates 9 intermediate tensors per call,
// any of which can trigger `musaMemMap` mid-capture.
void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps);

// Fused-add Gemma RMS norm: residual = input + residual (in-place), then
// input = (residual / RMS(residual)) * (weight + 1.0). Same fusion idea as
// the no-residual variant; lets callers skip the `(1.0 + weight)` tensor
// allocation entirely.
void fused_add_gemma_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                              torch::Tensor& residual,  // [..., hidden_size]
                              torch::Tensor& weight,    // [hidden_size]
                              double epsilon);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias,
                     std::optional<torch::Tensor> output_buf = std::nullopt);

// Fused split / reshape / cat for Qwen3.5 Gated-DeltaNet input projection.
// Ports sglang's TileLang `qkvzba_contiguous` kernel: one launch scatters the
// fused [Q|K|V|Z|B|A] projection into four pre-allocated output buffers.
//
// Required to keep the input-projection stage inside MUSA CUDA graphs, since
// the previous .contiguous() / cat path allocated four intermediate tensors
// per call and torch_musa rejects at::empty on a capture stream.
//
// Caller (Qwen3.5 GDN layer) owns persistent buffers for mixed_qkv / z / b / a
// and must size them for the largest decode bucket before capture. The
// function does NOT allocate.
void gdn_fused_qkvzba_split_contiguous(torch::Tensor fused,
                                       torch::Tensor mixed_qkv,
                                       torch::Tensor z,
                                       torch::Tensor b,
                                       torch::Tensor a,
                                       int64_t num_heads_qk,
                                       int64_t num_heads_v,
                                       int64_t head_qk,
                                       int64_t head_v);

// Fused single-token causal-conv1d decode update.
//
// Replaces the libtorch op chain (index_select -> .to(fp32) -> cat -> mul ->
// sum -> silu -> index_copy_) in the gdn_ops.cpp reference impl with a
// single CUDA kernel launch. Required for CUDA-graph capture on MUSA, where
// the libtorch chain triggers ~12 at::empty calls per call.
//
// Preconditions (graph-safe):
//   * x: [num_tokens, dim], num_tokens == batch (one token per sequence).
//   * weight: [dim, width], width in [2, 5].
//   * conv_state: [num_cache_lines, dim, state_len], state_len == width-1.
//     Updated in-place: ring shift + new x append.
//   * cache_indices: [batch], int32.
//   * output_buf: [num_tokens, dim], same dtype as x, stride(-1) == 1.
//     The kernel writes directly into this caller-owned buffer (no
//     allocations).
void causal_conv1d_decode_fused(const torch::Tensor& x,
                                const torch::Tensor& weight,
                                const std::optional<torch::Tensor>& bias,
                                torch::Tensor conv_state,
                                const torch::Tensor& cache_indices,
                                torch::Tensor output_buf,
                                int pad_slot_id,
                                bool silu_activation);

// Fused gated RMSNorm, single launch:
//   y[m, n] = (x[m, n] * rsqrt(mean(x[m,:]^2) + eps) * w[n])
//             * (z[m, n] * sigmoid(z[m, n]))
//
// Replaces the libtorch chain in cuda::gated_layer_norm_ref for the common
// Qwen3.5 case (RmsNormGated). Required for CUDA-graph capture: the ref impl
// allocates ~8 intermediate tensors per call (pow, mean, rsqrt, mul, sigmoid,
// mul, ...). Writes directly into a caller-owned output buffer; the call
// site checks `params.output_buf.has_value()` and that the layer config
// matches (is_rms_norm + norm_before_gate + z defined + single group + no
// bias). All compute happens in fp32 internally.
//
// Preconditions:
//   * x, z, output: 2D [M, N], same dtype/device, stride(-1) == 1.
//   * weight: 1D [N], same dtype as x, contiguous.
//   * dtype: fp32 / fp16 / bf16.
void gated_rms_norm_fused(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor& z,
                          torch::Tensor output,
                          double eps);

// CUDA-graph-safe partial rotary embedding (in-place).
//
// Rotates the first `rotary_dim` elements of every (token, head) slot in
// `query`/`key` in place, leaving the remaining `head_size - rotary_dim`
// elements unchanged. Replaces the libtorch reference in
// gdn_ops.cpp::partial_rotary_embedding which uses slice + .contiguous() +
// torch::cat -- all allocation primitives that torch_musa 2.7.1 rejects
// during stream capture. The underlying `rotary_embedding_kernel` already
// supports partial rotary natively (its inner loop only touches
// `num_heads * (rotary_dim / 2)` lanes).
//
// Preconditions:
//   * query/key: contiguous in the last dim, layout
//     `[num_tokens, num_heads * head_size]` or
//     `[num_tokens, num_heads, head_size]` (matching `rotary_embedding`).
//   * cos_sin_cache: 2D `[max_position, rotary_dim]`, contiguous.
//   * positions: int32 or int64.
//   * `rotary_dim` <= `head_size`, even.
void partial_rotary_embedding_inplace(torch::Tensor& positions,
                                      torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      int64_t head_size,
                                      int64_t rotary_dim,
                                      bool is_neox);

void cutlass_scaled_mm(torch::Tensor& c,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

// Static scaled FP8 quantization
// Quantizes input tensor to FP8 using a pre-computed scale factor
void static_scaled_fp8_quant(torch::Tensor& out,           // [..., d]
                             torch::Tensor const& input,   // [..., d]
                             torch::Tensor const& scale);  // [1]

// FP8 scaled quantize: quantizes input tensor to FP8 e4m3 format
// Returns: (quantized_output, scale)
std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& scale = std::nullopt);

// ============================================================================
// Fused RMSNorm + Static FP8 Quantization
// ============================================================================
// These functions combine RMSNorm and FP8 quantization to reduce memory
// bandwidth by avoiding the intermediate write-back to global memory.

// Fused RMSNorm + Static FP8 Quantization (without residual)
// Combines RMSNorm normalization and FP8 quantization in a single kernel.
// This is optimal for the first layer where no residual connection exists.
void rms_norm_static_fp8_quant(
    torch::Tensor& out,     // [..., hidden_size], FP8 output
    torch::Tensor& input,   // [..., hidden_size], input tensor
    torch::Tensor& weight,  // [hidden_size], RMSNorm weight
    torch::Tensor& scale,   // [1], FP8 quantization scale
    double epsilon);        // RMSNorm epsilon

// Fused Add + RMSNorm + Static FP8 Quantization (with residual)
// Combines residual addition, RMSNorm, and FP8 quantization in a single kernel.
// The residual tensor is updated in-place with the sum of input and residual.
void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size], FP8 output
    torch::Tensor& input,     // [..., hidden_size], input tensor
    torch::Tensor& residual,  // [..., hidden_size], residual (updated in-place)
    torch::Tensor& weight,    // [hidden_size], RMSNorm weight
    torch::Tensor& scale,     // [1], FP8 quantization scale
    double epsilon);          // RMSNorm epsilon

// FP8 scaled matmul for W8A8 quantization using CUTLASS kernels
// Performs: c = (a @ b.T) with scales applied
torch::Tensor fp8_scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device);

std::pair<torch::Tensor, torch::Tensor> compute_topk_general(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device);

torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures);

void fused_qk_norm_rope(
    torch::Tensor& qkv,   // Combined QKV tensor [num_tokens,
                          // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,  // Number of query heads
    int64_t num_heads_k,  // Number of key heads
    int64_t num_heads_v,  // Number of value heads
    int64_t head_dim,     // Dimension per head
    double eps,           // Epsilon for RMS normalization
    const torch::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    const torch::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    const torch::Tensor&
        cos_sin_cache,  // Cos/sin cache [max_position, rotary_dim]
    bool interleaved,   // Whether RoPE is applied in interleaved style
    const torch::Tensor& position_ids  // Position IDs for RoPE [num_tokens]
);

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func);

torch::Tensor random_sample(const torch::Tensor& probs);

torch::Tensor cutlass_fused_moe(
    const torch::Tensor& input,                   // [num_tokens, hidden]
    const torch::Tensor& token_selected_experts,  // [num_tokens, top_k]
    const torch::Tensor& token_final_scales,      // [num_tokens, top_k]
    const torch::Tensor&
        fc1_expert_weights,  // [num_experts, inter_dim, hidden]
    const torch::Tensor&
        fc2_expert_weights,  // [num_experts, hidden, inter_dim]
    torch::ScalarType output_dtype,
    const std::vector<torch::Tensor>& quant_scales,
    int32_t tp_size,
    int32_t tp_rank,
    int32_t ep_size,
    int32_t ep_rank,
    int32_t cluster_size,
    int32_t cluster_rank,
    const std::optional<torch::Tensor>& fc1_expert_biases = std::nullopt,
    const std::optional<torch::Tensor>& fc2_expert_biases = std::nullopt,
    const std::optional<torch::Tensor>& input_sf = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_alpha = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_beta = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_limit = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    bool enable_alltoall = false,
    bool use_deepseek_fp8_block_scale = false,
    bool use_w4_group_scaling = false,
    bool use_mxfp8_act_scaling = false,
    bool min_latency_mode = false,
    bool use_packed_weights = false,
    int32_t tune_max_num_tokens = 8192,
    ActivationType activation_type = ActivationType::SWIGLU);

// ---- moe_compute_index (moe_compute_index.cu) ----
// Fused routing index: bincount + argsort replacement.
// Returns {src_dst, dst_src, expert_sizes}.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts);

// ---- moe_combine_result (moe_combine.cu) ----
// Fused combine: reorder + weighted sum in one pass.
torch::Tensor moe_combine_result(const torch::Tensor& gemm2,
                                 const torch::Tensor& reduce_weight,
                                 int64_t N,
                                 int32_t topk);

}  // namespace xllm::kernel::cuda
