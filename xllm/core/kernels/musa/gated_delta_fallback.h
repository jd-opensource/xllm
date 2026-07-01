/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

// Pure-torch fallback implementations of the Qwen3.5 / Qwen3-Next gated
// delta-net kernels for the torch_musa (USE_CUDA + mcc_wrapper) build.
// Approach A: correctness first, performance later.

#pragma once

#include <torch/torch.h>

#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/param.h"

namespace xllm::kernel::cuda_fallback {

inline torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

inline torch::Tensor l2_norm(torch::Tensor& x, double eps) {
  return l2norm(x, -1, eps);
}

inline std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params) {
  auto a_f = params.a.to(torch::kFloat32);
  auto b_f = params.b.to(torch::kFloat32);
  auto dt_bias_f = params.dt_bias.to(torch::kFloat32);
  auto A_log_f = params.A_log.to(torch::kFloat32);
  auto beta = torch::sigmoid(b_f);
  auto sp = torch::nn::functional::softplus(
      a_f + dt_bias_f,
      torch::nn::functional::SoftplusFuncOptions()
          .beta(params.beta)
          .threshold(params.threshold));
  auto g = -(A_log_f.exp()) * sp;
  return {g.unsqueeze(0).contiguous(), beta.unsqueeze(0).contiguous()};
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params) {
  // Qwen3.5 gated-delta-net fused projection split (ported from xLLM MUSA ref).
  // mixed_qkvz is laid out PER K-HEAD GROUP (matching HF Qwen3NextGatedDeltaNet):
  //   view: [N, nk, 2*hk + 2*vpk*hv], per group: [q|k|v|z]
  // mixed_ba: [N, nk, 2*vpk] -> [b|a].
  const int64_t nk = params.num_heads_qk;
  const int64_t nv = params.num_heads_v;
  const int64_t hk = params.head_qk;
  const int64_t hv = params.head_v;
  const int64_t vpk = nv / nk;
  const int64_t per_group = 2 * hk + 2 * vpk * hv;

  const auto& qkvz = params.mixed_qkvz;
  const int64_t n = qkvz.numel() / qkvz.size(-1);
  auto qkvz_g = qkvz.reshape({n, nk, per_group});
  auto q = qkvz_g.slice(-1, 0, hk);
  auto k = qkvz_g.slice(-1, hk, 2 * hk);
  auto v = qkvz_g.slice(-1, 2 * hk, 2 * hk + vpk * hv);
  auto z = qkvz_g.slice(-1, 2 * hk + vpk * hv, per_group);

  auto q_flat = q.reshape({n, nk * hk}).contiguous();
  auto k_flat = k.reshape({n, nk * hk}).contiguous();
  auto v_flat = v.reshape({n, nv * hv}).contiguous();
  auto z_flat = z.reshape({n, nv * hv}).contiguous();

  auto mixed_qkv = torch::cat({q_flat, k_flat, v_flat}, -1).contiguous();

  const auto& ba = params.mixed_ba;
  auto ba_g = ba.reshape({n, nk, 2 * vpk});
  auto b = ba_g.slice(-1, 0, vpk).reshape({n, nv}).contiguous();
  auto a = ba_g.slice(-1, vpk, 2 * vpk).reshape({n, nv}).contiguous();

  return {mixed_qkv, z_flat, b, a};
}

// Depthwise causal conv1d, varlen prefill path (run_mode == 0).
//   x: [tokens, dim]; weight: [dim, width]; conv_state: [num_slots, width-1, dim]
inline torch::Tensor causal_conv1d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    const torch::IntArrayRef /*num_accepted_tokens_opt*/,
    int64_t activation_mode,
    int64_t /*pad_slot_id*/,
    int64_t /*run_mode*/) {
  // Varlen prefill: conv weight is [width, dim] (the layer transposes the loaded
  // [dim, width] weight back before calling this); conv_state is
  // [num_slots, width-1, dim]; x is [tokens, dim].
  const int64_t dim = x.size(-1);
  const int64_t width = weight.size(0);
  const int64_t state_len = width - 1;
  auto x_f = x.to(torch::kFloat32);
  auto w_f = weight.to(torch::kFloat32);
  torch::Tensor bias_f;
  const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
  if (has_bias) {
    bias_f = bias_opt.value().to(torch::kFloat32);
  }
  auto output = torch::empty_like(x_f);
  const int64_t num_seq = static_cast<int64_t>(query_start_loc_opt.size()) - 1;
  for (int64_t s = 0; s < num_seq; ++s) {
    const int64_t start = query_start_loc_opt[s];
    const int64_t end = query_start_loc_opt[s + 1];
    const int64_t seq_len = end - start;
    if (seq_len <= 0) {
      continue;
    }
    const int64_t slot = cache_indices_opt.empty() ? s : cache_indices_opt[s];
    const bool has_init =
        !initial_state_mode_opt.empty() && initial_state_mode_opt[s] != 0;
    auto seq = x_f.narrow(0, start, seq_len);
    torch::Tensor prefix;
    if (has_init) {
      prefix = conv_state[slot].to(torch::kFloat32);
    } else {
      prefix = torch::zeros({state_len, dim}, x_f.options());
    }
    auto padded = torch::cat({prefix, seq}, 0);
    auto out_seq = torch::zeros({seq_len, dim}, x_f.options());
    for (int64_t j = 0; j < width; ++j) {
      auto shifted = padded.narrow(0, j, seq_len);
      out_seq += shifted * w_f.select(0, j).unsqueeze(0);  // w_f[j] = [dim]
    }
    if (has_bias) {
      out_seq += bias_f.unsqueeze(0);
    }
    if (activation_mode != 0) {
      out_seq = torch::silu(out_seq);
    }
    output.narrow(0, start, seq_len).copy_(out_seq);
    auto tail = padded.narrow(0, padded.size(0) - state_len, state_len);
    conv_state[slot].copy_(tail.to(conv_state.dtype()));
  }
  return output.to(x.dtype());
}

// Decode-step causal conv1d update.
//   params.x: [tokens, dim]; params.conv_state: [num_slots, dim, width-1];
//   params.weight: [dim, width]
inline torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params) {
  auto x_f = params.x.to(torch::kFloat32);
  auto w_f = params.weight.to(torch::kFloat32);
  const int64_t tokens = x_f.size(0);
  const int64_t dim = x_f.size(-1);
  const int64_t width = w_f.size(-1);
  const int64_t state_len = width - 1;
  torch::Tensor bias_f;
  const bool has_bias = params.bias.has_value() && params.bias.value().defined();
  if (has_bias) {
    bias_f = params.bias.value().to(torch::kFloat32);
  }
  auto conv_state = params.conv_state;
  auto output = torch::empty({tokens, dim}, x_f.options());
  torch::Tensor indices;
  if (params.conv_state_indices.has_value()) {
    indices = params.conv_state_indices.value().to(torch::kCPU).to(torch::kLong);
  }
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t slot = indices.defined() ? indices[t].item<int64_t>() : t;
    auto state = conv_state[slot].to(torch::kFloat32);
    auto x_t = x_f[t];
    auto window = torch::cat({state, x_t.unsqueeze(-1)}, -1);
    auto out_t = torch::sum(window * w_f, -1);
    if (has_bias) {
      out_t += bias_f;
    }
    if (params.activation) {
      out_t = torch::silu(out_t);
    }
    output[t] = out_t;
    auto new_state = window.narrow(-1, 1, state_len).contiguous();
    conv_state[slot].copy_(new_state.to(conv_state.dtype()));
  }
  return output.to(params.x.dtype());
}

namespace detail {

inline torch::Tensor repeat_heads(const torch::Tensor& t,
                                  int64_t target_heads,
                                  int64_t head_dim) {
  const int64_t cur = t.size(head_dim);
  if (cur == target_heads) {
    return t;
  }
  const int64_t repeats = target_heads / cur;
  std::vector<int64_t> view_shape = t.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> out_shape = t.sizes().vec();
  out_shape[head_dim] = target_heads;
  return t.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(out_shape)
      .contiguous();
}

// Single-sequence recurrent gated delta rule. q,k,v,g,beta: [B,T,H,*].
// Returns {core_attn_out [B,T,Hv,Dv], last_state [B,Hv,K,V]}.
inline std::tuple<torch::Tensor, torch::Tensor> recurrent_one(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool use_qk_l2norm_in_kernel) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto tf = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = tf(query);
  key = tf(key);
  value = tf(value);
  beta = tf(beta);
  g = tf(g);
  const int64_t vh = value.size(1);
  query = repeat_heads(query, vh, 1);
  key = repeat_heads(key, vh, 1);
  int64_t bsz = key.size(0), nh = key.size(1), sl = key.size(2);
  int64_t kd = key.size(3), vd = value.size(3);
  float scale_val = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale_val;
  auto core = torch::zeros(
      {bsz, nh, sl, vd},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor state;
  if (!initial_state.has_value()) {
    state = torch::zeros(
        {bsz, nh, kd, vd},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    state = initial_state.value().to(value.device(), torch::kFloat32);
  }
  for (int64_t i = 0; i < sl; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    auto beta_t = beta.select(2, i).unsqueeze(-1);
    state = state * g_t;
    auto kv_mem = torch::sum(state * k_t.unsqueeze(-1), -2);
    auto delta = (v_t - kv_mem) * beta_t;
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core.select(2, i) = torch::sum(state * q_t.unsqueeze(-1), -2);
  }
  core = core.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core, state);
}

}  // namespace detail

namespace detail {

// Single-sequence chunked gated delta rule. q,k,v,g,beta: [B,T,H,*].
inline std::tuple<torch::Tensor, torch::Tensor> chunk_one(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    int64_t chunk_size,
    bool use_qk_l2norm_in_kernel) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto tf = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = tf(query);
  key = tf(key);
  value = tf(value);
  beta = tf(beta);
  g = tf(g);
  const int64_t vh = value.size(1);
  query = repeat_heads(query, vh, 1);
  key = repeat_heads(key, vh, 1);
  int64_t bsz = query.size(0), nh = query.size(1), sl = query.size(2);
  int64_t kd = key.size(-1), vd = value.size(-1);
  int64_t pad = (chunk_size - sl % chunk_size) % chunk_size;
  namespace F = torch::nn::functional;
  query = F::pad(query, F::PadFuncOptions({0, 0, 0, pad}));
  key = F::pad(key, F::PadFuncOptions({0, 0, 0, pad}));
  value = F::pad(value, F::PadFuncOptions({0, 0, 0, pad}));
  beta = F::pad(beta, F::PadFuncOptions({0, pad}));
  g = F::pad(g, F::PadFuncOptions({0, pad}));
  int64_t tsl = sl + pad;
  float scale = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);
  auto to_chunks = [chunk_size](torch::Tensor x) {
    auto sh = x.sizes();
    return x.reshape({sh[0], sh[1], sh[2] / chunk_size, chunk_size, sh[3]});
  };
  query = to_chunks(query);
  key = to_chunks(key);
  value = to_chunks(value);
  k_beta = to_chunks(k_beta);
  v_beta = to_chunks(v_beta);
  auto gs = g.sizes();
  g = g.reshape({gs[0], gs[1], gs[2] / chunk_size, chunk_size});
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
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone().contiguous();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
    auto row_sub_sum = (row.unsqueeze(-1).contiguous() * sub).contiguous().sum(-2).contiguous();
    auto row_final = (row + row_sub_sum).contiguous();
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }
  attn = attn + torch::eye(chunk_size,
                           torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
  torch::Tensor state;
  if (!initial_state.has_value()) {
    state = torch::zeros({bsz, nh, kd, vd},
                         torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  } else {
    state = initial_state.value().to(value);
  }
  auto core = torch::zeros_like(value);
  mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  int64_t num_chunks = tsl / chunk_size;
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i = (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
                      .masked_fill_(mask, 0.0);
    auto v_prime = torch::matmul(k_cumdecay.select(2, i), state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(), state);
    core.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    state = state * g_i_last.unsqueeze(-1).exp() + torch::matmul(k_g_exp, v_new);
  }
  auto cs = core.sizes();
  core = core.reshape({cs[0], cs[1], cs[2] * cs[3], cs[4]});
  core = core.slice(2, 0, sl);
  core = core.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core, state);
}

}  // namespace detail

// Packed/varlen chunked gated delta rule.
//   q,k,v: [1, total_tokens, H, *]; g,beta: [1, total_tokens, Hv]
//   initial_state: [num_seq, Hv, K, V]; cu_seqlens: [num_seq + 1]
// Returns {core_attn_out [1, total_tokens, Hv, Dv], last_state [num_seq,Hv,K,V]}.
inline std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
  auto cu = params.cu_seqlens.value().to(torch::kCPU).to(torch::kLong);
  const int64_t num_seq = cu.size(0) - 1;
  std::vector<torch::Tensor> cores;
  std::vector<torch::Tensor> states;
  cores.reserve(num_seq);
  states.reserve(num_seq);
  for (int64_t i = 0; i < num_seq; ++i) {
    const int64_t start = cu[i].item<int64_t>();
    const int64_t end = cu[i + 1].item<int64_t>();
    const int64_t len = end - start;
    auto q_i = params.q.narrow(1, start, len);
    auto k_i = params.k.narrow(1, start, len);
    auto v_i = params.v.narrow(1, start, len);
    auto g_i = params.g.narrow(1, start, len);
    auto beta_i = params.beta.narrow(1, start, len);
    std::optional<torch::Tensor> init;
    if (params.initial_state.has_value()) {
      init = params.initial_state.value().narrow(0, i, 1);
    }
    auto [core_i, state_i] = detail::chunk_one(
        q_i, k_i, v_i, g_i, beta_i, init, params.chunk_size,
        params.use_qk_l2norm_in_kernel);
    cores.push_back(core_i);
    states.push_back(state_i);
  }
  auto core = torch::cat(cores, 1).contiguous();
  auto last_state = torch::cat(states, 0).contiguous();
  return {core, last_state};
}

// Decode fused sigmoid gating + recurrent delta-rule update (FLA layout).
//   q,k,v: [B, T, Hv, *]; a,b: [B, T, Hv]; initial_state_source: ssm cache
//   [num_slots, Hv, K, V]; initial_state_indices: [B]. Updates cache in place.
// Returns core_attn_out [B, T, Hv, Dv].
inline torch::Tensor fused_sigmoid_gating_delta_rule_update(
    FusedSigmoidGatingDeltaRuleUpdateParams& params) {
  auto a_f = params.a.to(torch::kFloat32);
  auto b_f = params.b.to(torch::kFloat32);
  auto dt_bias_f = params.dt_bias.to(torch::kFloat32);
  auto A_log_f = params.A_log.to(torch::kFloat32);
  auto beta = torch::sigmoid(b_f);
  auto sp = torch::nn::functional::softplus(
      a_f + dt_bias_f,
      torch::nn::functional::SoftplusFuncOptions()
          .beta(params.softplus_beta)
          .threshold(params.softplus_threshold));
  auto g = (-(A_log_f.exp()) * sp).to(params.q.dtype());
  beta = beta.to(params.q.dtype());

  auto idx = params.initial_state_indices.to(torch::kCPU).to(torch::kLong);
  const int64_t bsz = params.q.size(0);
  auto& ssm = params.initial_state_source;
  std::vector<torch::Tensor> cores;
  cores.reserve(bsz);
  for (int64_t i = 0; i < bsz; ++i) {
    const int64_t slot = idx[i].item<int64_t>();
    auto state = ssm.narrow(0, slot, 1).to(torch::kFloat32);  // [1,Hv,K,V]
    auto q_i = params.q.narrow(0, i, 1);
    auto k_i = params.k.narrow(0, i, 1);
    auto v_i = params.v.narrow(0, i, 1);
    auto g_i = g.narrow(0, i, 1);
    auto beta_i = beta.narrow(0, i, 1);
    auto [core_i, state_i] = detail::recurrent_one(
        q_i, k_i, v_i, g_i, beta_i, state, params.use_qk_l2norm_in_kernel);
    ssm.narrow(0, slot, 1).copy_(state_i.to(ssm.dtype()));
    cores.push_back(core_i);
  }
  return torch::cat(cores, 0).contiguous();
}

// Non-FLA decode recurrent delta rule.
//   query/key/value: [tokens, H, D]; state: ssm cache [num_slots, Hv, K, V];
//   beta/g: [tokens, Hv]; actual_seq_lengths: [num_seq + 1] (leading 0 + per-seq
//   lengths); ssm_state_indices: [num_seq]. Updates cache; returns [tokens,Hv,Dv].
inline torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> /*scale*/,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& /*num_accepted_tokens*/,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& /*gk*/) {
  auto lens = actual_seq_lengths.value().to(torch::kCPU).to(torch::kLong);
  auto offsets = lens.cumsum(0);
  const int64_t num_seq = lens.size(0) - 1;
  auto idx = ssm_state_indices.value().to(torch::kCPU).to(torch::kLong);
  std::vector<torch::Tensor> cores;
  cores.reserve(num_seq);
  for (int64_t i = 0; i < num_seq; ++i) {
    const int64_t start = offsets[i].item<int64_t>();
    const int64_t end = offsets[i + 1].item<int64_t>();
    const int64_t len = end - start;
    if (len <= 0) {
      continue;
    }
    const int64_t slot = idx[i].item<int64_t>();
    auto st = state.narrow(0, slot, 1).to(torch::kFloat32);
    auto q_i = query.narrow(0, start, len).unsqueeze(0);
    auto k_i = key.narrow(0, start, len).unsqueeze(0);
    auto v_i = value.narrow(0, start, len).unsqueeze(0);
    auto g_i = g.value().narrow(0, start, len).unsqueeze(0);
    auto beta_i = beta.value().narrow(0, start, len).unsqueeze(0);
    auto [core_i, state_i] =
        detail::recurrent_one(q_i, k_i, v_i, g_i, beta_i, st, true);
    state.narrow(0, slot, 1).copy_(state_i.to(state.dtype()));
    cores.push_back(core_i.squeeze(0));
  }
  return torch::cat(cores, 0).contiguous();
}

// Packed/varlen recurrent gated delta rule. Mirrors chunk_gated_delta_rule I/O.
inline std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
  auto cu = params.cu_seqlens.value().to(torch::kCPU).to(torch::kLong);
  const int64_t num_seq = cu.size(0) - 1;
  std::vector<torch::Tensor> cores;
  std::vector<torch::Tensor> states;
  for (int64_t i = 0; i < num_seq; ++i) {
    const int64_t start = cu[i].item<int64_t>();
    const int64_t end = cu[i + 1].item<int64_t>();
    const int64_t len = end - start;
    auto q_i = params.q.narrow(1, start, len);
    auto k_i = params.k.narrow(1, start, len);
    auto v_i = params.v.narrow(1, start, len);
    auto g_i = params.g.narrow(1, start, len);
    auto beta_i = params.beta.value().narrow(1, start, len);
    std::optional<torch::Tensor> init;
    if (params.initial_state.has_value()) {
      init = params.initial_state.value().narrow(0, i, 1);
    }
    auto [core_i, state_i] = detail::recurrent_one(
        q_i, k_i, v_i, g_i, beta_i, init, params.use_qk_l2norm_in_kernel);
    cores.push_back(core_i);
    states.push_back(state_i);
  }
  return {torch::cat(cores, 1).contiguous(), torch::cat(states, 0).contiguous()};
}


// Gemma-style RMSNorm: y = x * rsqrt(mean(x^2)+eps) * (1 + gamma).
// Normalizes over the trailing gamma.dim() dimensions. Writes norm_out/rstd_out.
inline void gemma_rms_norm(GemmaRMSNormParams& params) {
  const auto& x = params.x;
  const int64_t dg = params.gamma.dim();
  std::vector<int64_t> dims;
  dims.reserve(dg);
  for (int64_t i = x.dim() - dg; i < x.dim(); ++i) {
    dims.push_back(i);
  }
  auto xf = x.to(torch::kFloat32);
  auto var = torch::mean(torch::square(xf), dims, /*keepdim=*/true);
  auto rstd = torch::rsqrt(var + params.epsilon);
  auto gamma_f = params.gamma.to(torch::kFloat32);
  auto y = xf * rstd * (1.0 + gamma_f);
  params.rstd_out = rstd;
  params.norm_out = y.to(x.dtype());
}


// Gated (grouped) RMS/Layer norm reference, ported from the xLLM MUSA build.
inline torch::Tensor gated_layer_norm(GatedLayerNormParams& params) {
  const auto x_shape_og = params.x.sizes();
  const int64_t last_dim = params.x.size(-1);
  auto x_2d = params.x.reshape({-1, last_dim});
  const int64_t M = x_2d.size(0);
  const int64_t N = x_2d.size(1);
  const int64_t group_size_val =
      params.group_size > 0 ? params.group_size : last_dim;
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
    x_norm_flat = torch::layer_norm(x_grouped_flat, {group_size_val},
                                    torch::Tensor(), torch::Tensor(),
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

}  // namespace xllm::kernel::cuda_fallback
