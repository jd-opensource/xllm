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

// M1 bring-up for the Qwen3.5 CUDA/MUSA GDN op set.
//
// Correctness-first libtorch reference kernels (same numerics as the FLA /
// torch_npu_ops references). Per-chunk recurrent scan and WY prep both run
// through torch ops on this backend; the matmul-heavy stages dispatch to
// MUSA's tensor-core path which is significantly faster than any
// hand-written kernel we measured for these shapes (Qwen3.5: chunk=64,
// K=V=128).
#include "gdn_ops.h"

#include <glog/logging.h>

#include "core/common/macros.h"
#include "core/kernels/param.h"
#include "cuda_ops_api.h"
#include "mate_gdn_ops.h"
#include "utils.h"

namespace xllm {
namespace kernel {
namespace cuda {

namespace {

torch::Tensor as_long_indices(const torch::Tensor& indices) {
  if (indices.scalar_type() == torch::kLong) {
    return indices.is_contiguous() ? indices : indices.contiguous();
  }
  return indices.to(torch::kLong).contiguous();
}

}  // namespace

namespace {

inline torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

// Gated RMS/LayerNorm reference (matches torch_npu_ops/triton_layernorm_fwd_test).
torch::Tensor gated_layer_norm_ref(GatedLayerNormParams& params) {
  const auto x_shape_og = params.x.sizes();
  const int64_t last_dim = params.x.size(-1);
  auto x_2d = params.x.reshape({-1, last_dim});
  const int64_t M = x_2d.size(0);
  const int64_t N = x_2d.size(1);
  const int64_t group_size_val =
      params.group_size > 0 ? params.group_size : last_dim;
  TORCH_CHECK(N % group_size_val == 0,
              "gated_layer_norm: N must be divisible by group_size");
  const int64_t ngroups = N / group_size_val;

  torch::Tensor z_2d;
  if (params.z.has_value() && params.z.value().defined()) {
    z_2d = params.z.value().reshape({-1, last_dim});
  }

  torch::Tensor x_input = x_2d;
  if (z_2d.defined() && !params.norm_before_gate) {
    x_input = x_2d * (z_2d * torch::sigmoid(z_2d));
  }

  auto x_grouped = x_input.unfold(1, group_size_val, group_size_val);
  auto x_grouped_flat = x_grouped.reshape({-1, group_size_val});

  torch::Tensor x_norm_flat;
  if (!params.is_rms_norm) {
    x_norm_flat = torch::layer_norm(
        x_grouped_flat,
        {group_size_val},
        torch::Tensor(),
        torch::Tensor(),
        params.eps);
  } else {
    auto mean_sq = x_grouped_flat.pow(2).mean(-1, /*keepdim=*/true);
    x_norm_flat = x_grouped_flat * torch::rsqrt(mean_sq + params.eps);
  }

  auto x_norm = x_norm_flat.reshape({M, ngroups, group_size_val})
                    .contiguous()
                    .view({M, N});
  auto y = x_norm * params.weight.to(x_norm.dtype());
  if (params.bias.defined()) {
    y = y + params.bias.to(y.dtype());
  }
  if (z_2d.defined() && params.norm_before_gate) {
    y = y * (z_2d.to(y.dtype()) * torch::sigmoid(z_2d.to(y.dtype())));
  }
  return y.reshape(x_shape_og);
}

// One recurrent GDN step on state shaped [..., H, K, V] (batched or not).
inline torch::Tensor recurrent_gdn_step(torch::Tensor& state,
                                          const torch::Tensor& q_t,
                                          const torch::Tensor& k_t,
                                          const torch::Tensor& v_t,
                                          const torch::Tensor& g_t,
                                          const torch::Tensor& beta_t) {
  auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
  state.mul_(g_exp);
  auto kv_mem = (state * k_t.unsqueeze(-1)).sum(-2);
  auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
  state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
  return (state * q_t.unsqueeze(-1)).sum(-2);
}

}  // namespace

torch::Tensor l2_norm(torch::Tensor& x, double eps) {
  // L2 normalize the last dimension: y = x / sqrt(sum(x^2, -1) + eps).
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params) {
  // Qwen3.5 / Qwen3-Next gated-delta-rule gating (matches FLA reference):
  //   g    = -exp(A_log)[None, :] * softplus(a + dt_bias[None, :],
  //                                          beta=beta, threshold=threshold)
  //   beta = sigmoid(b)
  // A_log, dt_bias: [num_heads]; a, b: [B, num_heads].
  // Outputs g, beta: same shape as a/b. Computation in float32 for stability.
  const auto& A_log = params.A_log;
  const auto& a = params.a;
  const auto& b = params.b;
  const auto& dt_bias = params.dt_bias;
  const auto orig_dtype = a.scalar_type();

  // Fused single-kernel path (collapses the ~10 torch ops below into one
  // elementwise launch). Falls back to the torch reference for dtypes the
  // kernel does not handle.
  if (!a.is_cpu() &&
      (orig_dtype == torch::kFloat32 || orig_dtype == torch::kBFloat16) &&
      b.scalar_type() == orig_dtype) {
    return gdn_gating(a, b, A_log, dt_bias, params.beta, params.threshold);
  }

  auto a_f32 = a.to(torch::kFloat32);
  auto b_f32 = b.to(torch::kFloat32);
  auto A_log_f32 = A_log.to(torch::kFloat32);
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32);

  // softplus(x; beta, threshold) = x if beta*x > threshold else
  //                                (1/beta) * log1p(exp(beta * x))
  auto pre = a_f32 + dt_bias_f32.unsqueeze(0);
  auto sp = torch::nn::functional::softplus(
      pre,
      torch::nn::functional::SoftplusFuncOptions()
          .beta(params.beta)
          .threshold(params.threshold));
  auto g_f32 = -torch::exp(A_log_f32).unsqueeze(0) * sp;
  auto beta_f32 = torch::sigmoid(b_f32);

  return {g_f32.to(orig_dtype), beta_f32.to(orig_dtype)};
}

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  const auto initial_dtype = query.scalar_type();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  auto to_f32_bhtd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_bhtd(query);
  key = to_f32_bhtd(key);
  value = to_f32_bhtd(value);
  g = to_f32_bhtd(g);
  torch::Tensor beta_f32;
  if (params.beta.has_value() && params.beta.value().defined()) {
    beta_f32 = to_f32_bhtd(params.beta.value());
  } else {
    beta_f32 = torch::ones_like(g);
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);
  const float scale_val =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale_val;

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state =
        params.initial_state.value().to(torch::kFloat32).transpose(-1, -2);
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i);
    auto beta_t = beta_f32.select(2, i);
    auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
    last_recurrent_state.mul_(g_exp);
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
    last_recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
    core_attn_out.select(2, i) =
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  last_recurrent_state = last_recurrent_state.transpose(-1, -2);
  return {core_attn_out, last_recurrent_state};
}

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params) {
  auto x = params.x;
  auto weight = params.weight;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(1) == 1,
                "causal_conv1d_update: expected weight [dim, 1, width]");
    weight = weight.squeeze(1);
  }
  TORCH_CHECK(weight.dim() == 2,
              "causal_conv1d_update: expected weight [dim, width]");
  TORCH_CHECK(params.conv_state.dim() == 3,
              "causal_conv1d_update: expected conv_state "
              "[num_cache_lines, dim, state_len]");
  TORCH_CHECK(params.conv_state_indices.has_value(),
              "causal_conv1d_update: conv_state_indices is required");
  TORCH_CHECK(params.query_start_loc.has_value(),
              "causal_conv1d_update: query_start_loc is required");

  const int64_t dim = weight.size(0);
  const int64_t width = weight.size(1);
  const int64_t state_len = width - 1;
  TORCH_CHECK(params.conv_state.size(1) == dim,
              "causal_conv1d_update: conv_state dim mismatch");
  TORCH_CHECK(params.conv_state.size(2) == state_len,
              "causal_conv1d_update: conv_state state_len mismatch");

  const auto& cache_indices_raw = params.conv_state_indices.value();
  const int64_t batch = cache_indices_raw.size(0);
  const int64_t conv_num_tokens_pre = x.size(0);

  // CUDA-graph-safe fused fast path. Engaged when:
  //   * caller provided a persistent output_buf (graph mode warmed it up)
  //   * single-token-per-seq decode (num_tokens == batch)
  //   * state_len > 0 (width >= 2; Qwen3.5 uses width=4)
  //   * cache_indices is already int32 (no host-side conversion needed)
  //   * width in [2, 5] (supported by the fused kernel)
  // Skips all libtorch op allocations (weight.to(fp32), x.to(fp32),
  // index_select, cat, mul, sum, silu, index_copy_, ...) and writes the
  // result directly into the caller-owned buffer. The conv_state ring is
  // updated in-place by the kernel.
  if (params.output_buf.has_value() && params.output_buf->defined() &&
      conv_num_tokens_pre == batch && state_len > 0 && width >= 2 &&
      width <= 5 && cache_indices_raw.scalar_type() == torch::kInt32 &&
      cache_indices_raw.is_contiguous()) {
    causal_conv1d_decode_fused(x,
                               weight,
                               params.bias,
                               params.conv_state,
                               cache_indices_raw,
                               *params.output_buf,
                               static_cast<int>(params.pad_slot_id),
                               params.activation);
    return *params.output_buf;
  }

  auto weight_f32 = weight.to(torch::kFloat32);
  auto x_f32 = x.to(torch::kFloat32);
  auto out = torch::empty_like(x_f32);
  std::optional<torch::Tensor> bias_f32;
  if (params.bias.has_value() && params.bias.value().defined()) {
    bias_f32 = params.bias.value().to(torch::kFloat32);
  }

  const auto& cache_indices = cache_indices_raw.contiguous();
  const auto& query_start_loc = params.query_start_loc.value().contiguous();

  // Fast vectorized path for standard decode: one query token per sequence
  // (num_tokens == batch), state_len > 0, no padded slots (pad_slot_id=-1 never
  // matches a valid block id). Removes the per-row .item() host syncs and the
  // python-side token loop; all sequences are updated in parallel on-device.
  const int64_t conv_num_tokens = x_f32.size(0);
  if (conv_num_tokens == batch && state_len > 0) {
    auto idx = as_long_indices(cache_indices);
    auto history =
        params.conv_state.index_select(0, idx).to(torch::kFloat32);  // [B,dim,sl]
    auto window =
        torch::cat({history, x_f32.unsqueeze(-1)}, /*dim=*/-1);  // [B,dim,width]
    auto token_out = (window * weight_f32.unsqueeze(0)).sum(-1);  // [B,dim]
    if (bias_f32.has_value()) {
      token_out = token_out + bias_f32.value();
    }
    if (params.activation) {
      token_out = torch::silu(token_out);
    }
    // New state = window with the oldest column dropped (== [history[1:], x]).
    auto new_state = window.narrow(/*dim=*/-1, 1, state_len).contiguous();
    params.conv_state.index_copy_(
        0, idx, new_state.to(params.conv_state.scalar_type()));
    return token_out.to(x.scalar_type());
  }

  for (int64_t seq = 0; seq < batch; ++seq) {
    const int64_t cache_idx = cache_indices[seq].item<int64_t>();
    if (cache_idx == params.pad_slot_id) {
      continue;
    }
    const int64_t start = query_start_loc[seq].item<int64_t>();
    const int64_t end = query_start_loc[seq + 1].item<int64_t>();
    auto history =
        params.conv_state[cache_idx].to(torch::kFloat32).clone();  // [dim, state_len]

    for (int64_t token_idx = start; token_idx < end; ++token_idx) {
      auto x_t = x_f32[token_idx];
      torch::Tensor token_out;
      if (state_len == 0) {
        token_out = weight_f32.select(1, 0) * x_t;
      } else {
        auto window = torch::cat({history, x_t.unsqueeze(-1)}, /*dim=*/-1);
        token_out = (window * weight_f32).sum(-1);
      }
      if (bias_f32.has_value()) {
        token_out = token_out + bias_f32.value();
      }
      if (params.activation) {
        token_out = torch::silu(token_out);
      }
      out[token_idx] = token_out;
      if (state_len > 0) {
        if (state_len == 1) {
          history = x_t.unsqueeze(-1);
        } else {
          history = torch::cat(
              {history.slice(/*dim=*/-1, /*start=*/1, /*end=*/state_len),
               x_t.unsqueeze(-1)},
              /*dim=*/-1);
        }
      }
    }
    params.conv_state[cache_idx] = history.to(params.conv_state.scalar_type());
  }

  return out.to(x.scalar_type());
}

torch::Tensor gated_layer_norm(GatedLayerNormParams& params) {
  // CUDA-graph-safe fused fast path for the dominant Qwen3.5 RmsNormGated call
  // pattern (norm_before_gate=true RMSNorm with a sigmoid-gated z, single
  // group, no bias). Writes into a caller-owned `output_buf` so the kernel
  // host wrapper performs zero allocations under stream capture; the libtorch
  // ref impl below chains ~8 ops that each call at::empty.
  //
  // Falls back to the ref impl whenever any precondition isn't met (e.g.
  // bias provided, layer_norm rather than RMSNorm, non-trivial group_size,
  // gating skipped, unsupported dtype, caller didn't preallocate output_buf).
  if (params.output_buf.has_value() && params.output_buf->defined() &&
      params.is_rms_norm && params.norm_before_gate &&
      params.z.has_value() && params.z.value().defined() &&
      !params.bias.defined()) {
    const int64_t last_dim = params.x.size(-1);
    const int64_t group_size_val =
        params.group_size > 0 ? params.group_size : last_dim;
    const bool dtype_ok = params.x.scalar_type() == torch::kBFloat16 ||
                          params.x.scalar_type() == torch::kHalf ||
                          params.x.scalar_type() == torch::kFloat32;
    const torch::Tensor& z = params.z.value();
    // We require full contiguity on x/z/output_buf so that the row-major
    // 2D reshape below is guaranteed to be a view (i.e. reshape does NOT
    // synthesize a contiguous copy via at::empty, which torch_musa rejects
    // during stream capture).
    if (group_size_val == last_dim && dtype_ok &&
        params.x.scalar_type() == z.scalar_type() &&
        params.x.scalar_type() == params.weight.scalar_type() &&
        params.x.scalar_type() == params.output_buf->scalar_type() &&
        params.x.is_contiguous() && z.is_contiguous() &&
        params.output_buf->is_contiguous() && params.weight.is_contiguous() &&
        params.weight.dim() == 1) {
      auto x_2d = params.x.reshape({-1, last_dim});
      auto z_2d = z.reshape({-1, last_dim});
      auto out_2d = params.output_buf->reshape({-1, last_dim});
      gated_rms_norm_fused(x_2d, params.weight, z_2d, out_2d, params.eps);
      return params.output_buf->reshape(params.x.sizes());
    }
  }
  return gated_layer_norm_ref(params);
}

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params) {
  // CUDA-graph-safe path: the underlying rotary_embedding_kernel already
  // applies rotation only to the first `rotary_dim` elements of each head
  // (its inner loop iterates `num_heads * (rotary_dim / 2)` lanes), so we
  // can drive it in-place on the full Q/K tensors without materialising
  // q_rot/k_rot via `slice(-1, 0, rotary_dim).contiguous()` and then
  // re-concatenating with the pass-through suffix. Both `.contiguous()`
  // and `torch::cat` invoke `at::empty`-class allocators that torch_musa
  // 2.7.1 rejects during stream capture (this was the
  //   contiguous -> clone -> empty_like -> EmptyMUSA
  // trace observed during the decode-only graph capture).
  //
  // The caller (xllm::layer::PartialRotaryEmbeddingImpl::forward) writes
  // the returned tensors back into Q and K; passing the originals through
  // is semantically identical now that rotation is applied in place.
  partial_rotary_embedding_inplace(params.positions,
                                   params.query,
                                   params.key,
                                   params.cos_sin_cache,
                                   params.head_size,
                                   params.rotary_dim,
                                   params.is_neox_style);
  return {params.query, params.key};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params) {
  // Qwen3.5 gated-delta-net: split the fused projections back into the
  // four streams the rest of the pipeline expects.
  //
  // mixed_qkvz layout along the last dim: [Q | K | V | Z]
  //   Q : num_heads_qk * head_qk
  //   K : num_heads_qk * head_qk
  //   V : num_heads_v  * head_v
  //   Z : num_heads_v  * head_v
  // mixed_ba layout along the last dim: [B | A]
  //   B : num_heads_v
  //   A : num_heads_v
  //
  // Output:
  //   mixed_qkv : concat([Q, K, V], dim=-1)
  //   z         : the Z slice (caller reshapes to [..., num_heads_v, head_v])
  //   b, a      : per-head gates (caller reshapes to [..., num_heads_v])
  // mixed_qkvz is laid out PER K-HEAD GROUP (matching HF
  // Qwen3NextGatedDeltaNet.fix_query_key_value_ordering and the
  // merge_qkvz_from_split_activations producer), NOT as global Q|K|V|Z blocks:
  //   view: [N, num_heads_qk, 2*head_qk + 2*(num_heads_v/num_heads_qk)*head_v]
  //   per group g: [ q_g(head_qk) | k_g(head_qk) | v_g(vpk*head_v) | z_g(vpk*head_v) ]
  // De-interleave the groups and re-pack into the global [Q|K|V] blocks that
  // process_mixed_qkv expects (V/Z group-major so q/k repeat_interleave aligns).
  const int64_t nk = static_cast<int64_t>(params.num_heads_qk);
  const int64_t nv = static_cast<int64_t>(params.num_heads_v);
  const int64_t hk = static_cast<int64_t>(params.head_qk);
  const int64_t hv = static_cast<int64_t>(params.head_v);
  TORCH_CHECK(nk > 0 && nv > 0 && nv % nk == 0,
              "fused_qkvzba_split_reshape_cat: invalid head counts nk=",
              nk,
              " nv=",
              nv);
  const int64_t vpk = nv / nk;
  const int64_t per_group = 2 * hk + 2 * vpk * hv;

  const auto& qkvz = params.mixed_qkvz;
  const int64_t n = qkvz.numel() / qkvz.size(-1);
  TORCH_CHECK(
      qkvz.size(-1) == nk * per_group,
      "fused_qkvzba_split_reshape_cat: mixed_qkvz last dim mismatch, got ",
      qkvz.size(-1),
      " expected ",
      nk * per_group);

  auto qkvz_g = qkvz.reshape({n, nk, per_group});
  auto q = qkvz_g.slice(-1, 0, hk);
  auto k = qkvz_g.slice(-1, hk, 2 * hk);
  auto v = qkvz_g.slice(-1, 2 * hk, 2 * hk + vpk * hv);
  auto z = qkvz_g.slice(-1, 2 * hk + vpk * hv, per_group);

  const auto& ba = params.mixed_ba;
  TORCH_CHECK(ba.size(-1) == 2 * nv,
              "fused_qkvzba_split_reshape_cat: mixed_ba last dim mismatch, got ",
              ba.size(-1),
              " expected ",
              2 * nv);
  auto ba_g = ba.reshape({n, nk, 2 * vpk});
  auto b_slice = ba_g.slice(-1, 0, vpk);          // [n, nk, vpk] strided
  auto a_slice = ba_g.slice(-1, vpk, 2 * vpk);    // [n, nk, vpk] strided

  const int64_t qkv_dim = 2 * nk * hk + nv * hv;
  const int64_t z_dim = nv * hv;

  // CUDA-graph-safe path: caller pre-allocated grow-only persistent buffers.
  // Avoids the `reshape().contiguous() ... torch::cat()` chain that calls
  // `at::empty` -> `EmptyMUSA` and aborts MUSA stream capture with
  //   "operation not permitted when stream is capturing".
  // Each `copy_` writes a strided source view directly into a contiguous
  // slice of the persistent buffer, performing no host-side allocation.
  if (params.mixed_qkv_out_buf.defined() && params.z_out_buf.defined() &&
      params.b_out_buf.defined() && params.a_out_buf.defined() &&
      params.mixed_qkv_out_buf.size(0) >= n &&
      params.mixed_qkv_out_buf.size(1) == qkv_dim &&
      params.z_out_buf.size(0) >= n && params.z_out_buf.size(1) == z_dim &&
      params.b_out_buf.size(0) >= n && params.b_out_buf.size(1) == nv &&
      params.a_out_buf.size(0) >= n && params.a_out_buf.size(1) == nv) {
    auto mixed_qkv_buf =
        params.mixed_qkv_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto z_buf =
        params.z_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto b_buf =
        params.b_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto a_buf =
        params.a_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);

    // mixed_qkv layout: [n, q | k | v]
    //   q: nk * hk  ; k: nk * hk  ; v: nv * hv
    // Each section is filled by copying the corresponding strided view of
    // qkvz_g into the appropriate contiguous slice.
    mixed_qkv_buf.narrow(/*dim=*/1, /*start=*/0, /*length=*/nk * hk)
        .view({n, nk, hk})
        .copy_(q);
    mixed_qkv_buf.narrow(/*dim=*/1, /*start=*/nk * hk, /*length=*/nk * hk)
        .view({n, nk, hk})
        .copy_(k);
    mixed_qkv_buf
        .narrow(/*dim=*/1,
                /*start=*/2 * nk * hk,
                /*length=*/nv * hv)
        .view({n, nk, vpk * hv})
        .copy_(v);
    z_buf.view({n, nk, vpk * hv}).copy_(z);
    b_buf.view({n, nk, vpk}).copy_(b_slice);
    a_buf.view({n, nk, vpk}).copy_(a_slice);

    return {mixed_qkv_buf, z_buf, b_buf, a_buf};
  }

  auto q_flat = q.reshape({n, nk * hk}).contiguous();
  auto k_flat = k.reshape({n, nk * hk}).contiguous();
  auto v_flat = v.reshape({n, nv * hv}).contiguous();
  auto z_flat = z.reshape({n, nv * hv}).contiguous();

  auto mixed_qkv = torch::cat({q_flat, k_flat, v_flat}, -1).contiguous();

  auto b = b_slice.reshape({n, nv}).contiguous();
  auto a = a_slice.reshape({n, nv}).contiguous();

  return {mixed_qkv, z_flat, b, a};
}

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
  // Pure torch reference port of the FLA / Qwen3-Next chunk-GDN. Operates
  // chunk-by-chunk; expensive but numerically equivalent to the fused
  // kernels used on NPU. Inputs:
  //   q,k : [B, T, Hqk, K]
  //   v   : [B, T, H,   V]
  //   g,beta : [B, T, H]
  // Outputs:
  //   core_attn_out : [B, T, H, V] (same dtype as q on entry)
  //   last_recurrent_state : [B, H, K, V] in float32
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  auto beta = params.beta;
  const int64_t chunk_size = 64;
  const auto initial_dtype = query.dtype();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  // Handle GQA in the linear-attention path (Qwen3.5: Hqk=16, Hv=48):
  // q,k come in with Hqk heads, v with Hv heads. Expand q and k along the
  // head axis (dim=2 before transpose -> dim=1 after) so all four match Hv.
  const int64_t Hqk = query.size(2);
  const int64_t Hv = value.size(2);
  if (Hqk != Hv) {
    TORCH_CHECK(Hv % Hqk == 0,
                "chunk_gated_delta_rule: Hv (",
                Hv,
                ") must be a multiple of Hqk (",
                Hqk,
                ") for GQA expansion");
    const int64_t repeat = Hv / Hqk;
    query = query.repeat_interleave(repeat, /*dim=*/2);
    key = key.repeat_interleave(repeat, /*dim=*/2);
  }

  auto to_f32_thd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_thd(query);  // [B, H, T, K]
  key = to_f32_thd(key);
  value = to_f32_thd(value);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);  // [B, H, T]
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);

  const int64_t pad_size = (chunk_size - sequence_length % chunk_size) %
                           chunk_size;
  using PadOpts = torch::nn::functional::PadFuncOptions;
  if (pad_size != 0) {
    query = torch::nn::functional::pad(query, PadOpts({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, PadOpts({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, PadOpts({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, PadOpts({0, pad_size}));
    g = torch::nn::functional::pad(g, PadOpts({0, pad_size}));
  }
  const int64_t total_sequence_length = sequence_length + pad_size;
  const float scale =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);

  auto reshape_to_chunks = [chunk_size](const torch::Tensor& x) {
    return x.reshape({x.size(0), x.size(1), x.size(2) / chunk_size, chunk_size,
                      x.size(3)});
  };
  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);
  g = g.reshape({g.size(0), g.size(1), g.size(2) / chunk_size, chunk_size});

  auto mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);
  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32).tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) attn = attn.contiguous();
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }
  attn = attn + torch::eye(chunk_size, torch::TensorOptions()
                                           .dtype(attn.dtype())
                                           .device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, k_beta * g.exp().unsqueeze(-1));

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state = params.initial_state.value().to(value.dtype());
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  }
  auto core_attn_out = torch::zeros_like(value);
  const int64_t num_chunks = total_sequence_length / chunk_size;

  // Per-chunk recurrent scan via torch matmuls. An A/B against a hand-written
  // CUDA kernel showed the torch path is ~42x faster on mp31 because each
  // matmul hits MUSA's tensor-core path while the kernel was register-serial.
  auto upper_mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions()
                      .dtype(torch::kBool)
                      .device(query.device())),
      1);
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(upper_mask, 0.0);
    auto v_prime =
        torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  const auto s = core_attn_out.sizes();
  core_attn_out = core_attn_out.reshape({s[0], s[1], s[2] * s[3], s[4]});
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}

torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& /*num_accepted_tokens*/,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& /*gk*/) {
  CHECK(scale.has_value()) << "recurrent_gated_delta_rule requires scale";
  CHECK(g.has_value()) << "recurrent_gated_delta_rule requires g";
  CHECK(beta.has_value()) << "recurrent_gated_delta_rule requires beta";

  const auto orig_dtype = value.scalar_type();
  auto q = query.to(torch::kFloat32);
  auto k = key.to(torch::kFloat32);
  auto v = value.to(torch::kFloat32);

  // GQA: q/k may have fewer heads than v/g/beta (Qwen3.5: Hqk=16, Hv=32).
  // Match chunk_gated_delta_rule by expanding q/k along the head axis.
  const int64_t Hqk = q.size(1);
  const int64_t Hv = v.size(1);
  if (Hqk != Hv) {
    TORCH_CHECK(Hv % Hqk == 0,
                "recurrent_gated_delta_rule: Hv (",
                Hv,
                ") must be a multiple of Hqk (",
                Hqk,
                ") for GQA expansion");
    const int64_t repeat = Hv / Hqk;
    q = q.repeat_interleave(repeat, /*dim=*/1);
    k = k.repeat_interleave(repeat, /*dim=*/1);
  }
  auto g_in = g.value().to(torch::kFloat32);
  auto beta_in = beta.value().to(torch::kFloat32);
  const double sc = scale.value();
  const double l2_eps = 1e-6;

  const int64_t num_tokens = q.size(0);
  auto out = torch::empty({num_tokens, q.size(1), v.size(-1)},
                          q.options().dtype(torch::kFloat32));

  int64_t batch_size = num_tokens;
  if (ssm_state_indices.has_value()) {
    batch_size = ssm_state_indices.value().size(0);
  }

  // Fused fast path: the hand-written MUSA kernel does gather + qk L2-norm +
  // q-scale + recurrent step + scatter in a single launch (state updated in
  // place). Requires standard decode (one token/seq), square head dims (K==V),
  // and a float/bf16 contiguous ssm cache.
  // (Approach A) kernel fast-path disabled; use torch reference below.

  // Non-kernel paths apply the qk L2-norm + q-scale that the fused kernel folds
  // in internally (the layer no longer pre-normalizes on CUDA).
  q = l2_norm(q, l2_eps) * static_cast<float>(sc);
  k = l2_norm(k, l2_eps);

  if (ssm_state_indices.has_value() && num_tokens == batch_size) {
    auto idx = as_long_indices(ssm_state_indices.value());
    // ssm_cache stores [*, H, V, K]; working state is [B, H, K, V].
    auto st = state.index_select(0, idx)
                  .to(torch::kFloat32)
                  .transpose(-1, -2)
                  .contiguous();
    auto g_exp = g_in.exp().unsqueeze(-1).unsqueeze(-1);    // [B,H,1,1]
    st = st * g_exp;
    auto kv_mem = (st * k.unsqueeze(-1)).sum(-2);           // [B,H,V]
    auto delta = (v - kv_mem) * beta_in.unsqueeze(-1);      // [B,H,V]
    st = st + k.unsqueeze(-1) * delta.unsqueeze(-2);        // [B,H,K,V]
    auto out_v = (st * q.unsqueeze(-1)).sum(-2);            // [B,H,V]
    state.index_copy_(
        0, idx, st.transpose(-1, -2).contiguous().to(state.scalar_type()));
    return out_v.to(orig_dtype);
  }

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t start = b;
    int64_t end = b + 1;
    if (actual_seq_lengths.has_value()) {
      start = actual_seq_lengths.value()[b].item<int64_t>();
      end = actual_seq_lengths.value()[b + 1].item<int64_t>();
    }

    int64_t slot = b;
    if (ssm_state_indices.has_value()) {
      slot = ssm_state_indices.value()[b].item<int64_t>();
    }

    // ssm_cache stores [H, V, K]; working state is [H, K, V].
    auto st = state[slot].to(torch::kFloat32).transpose(-1, -2).contiguous();

    for (int64_t t = start; t < end; ++t) {
      out[t] = recurrent_gdn_step(
          st, q[t], k[t], v[t], g_in[t], beta_in[t]);
    }

    state[slot] = st.transpose(-1, -2).to(state.scalar_type());
  }

  return out.to(orig_dtype);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
