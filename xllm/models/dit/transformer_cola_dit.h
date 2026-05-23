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

#include <cmath>
#include <vector>

namespace {
// Drop-in replacement for nn::Linear::forward that calls torch::linear
// (i.e. at::linear / F.linear), which routes through the same ATen dispatch
// path as Python F.linear.  This guarantees bit-identical matmul results
// compared to the reference PyTorch implementation regardless of which
// cuBLAS workspace the LibTorch nn::Linear backend might select.
inline torch::Tensor torch_linear(const torch::nn::Linear& linear,
                                  const torch::Tensor& input) {
  return torch::linear(input, linear->weight, linear->bias);
}
}  // namespace

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/layers/common/linear.h"
#include "models/model_registry.h"
#include "transformer_longcat_audiodit.h"  // load_module_from_state_dicts

namespace xllm {

// ---------------------------------------------------------------------------
// Sinusoidal Timestep Embedding
// ---------------------------------------------------------------------------

// Matches diffusers convention: flip_sin_to_cos=False, downscale_freq_shift=0.
// Denominator is half_dim, NOT half_dim-1.
inline torch::Tensor get_sinusoidal_embedding(const torch::Tensor& timesteps,
                                              int64_t embedding_dim) {
  int64_t half_dim = embedding_dim / 2;
  // exponent[j] = -log(10000) * j / half_dim  (log of frequency)
  auto exponent = -std::log(10000.0f) *
                  torch::arange(0,
                                half_dim,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(timesteps.device())) /
                  static_cast<float>(half_dim);
  // freq[j] = exp(exponent[j]) = 10000^(-j/half_dim)
  // emb[t, j] = t * freq[j]  — matches Python: emb = torch.exp(exponent); emb =
  // timesteps * emb
  auto emb = timesteps.to(torch::kFloat32).unsqueeze(1) *
             torch::exp(exponent).unsqueeze(0);
  return torch::cat({torch::sin(emb), torch::cos(emb)}, /*dim=*/-1);
}

// ---------------------------------------------------------------------------
// TimestepEmbedding: sinusoidal -> 3-layer MLP (SiLU)
// ---------------------------------------------------------------------------

class ColaTimestepEmbeddingImpl : public torch::nn::Module {
 public:
  ColaTimestepEmbeddingImpl(int64_t sinusoidal_dim,
                            int64_t hidden_dim,
                            int64_t output_dim) {
    proj_in_ = register_module("proj_in",
                               torch::nn::Linear(sinusoidal_dim, hidden_dim));
    proj_hid_ =
        register_module("proj_hid", torch::nn::Linear(hidden_dim, hidden_dim));
    proj_out_ =
        register_module("proj_out", torch::nn::Linear(hidden_dim, output_dim));
    act_ = register_module("act", torch::nn::SiLU());
  }

  torch::Tensor forward(const torch::Tensor& timestep) {
    // Match official Python TimestepEmbedding.forward():
    //   emb = _get_sinusoidal_embedding(timestep, ...).to(dtype)
    // The .to(dtype) cast is critical: it forces the float32 sinusoidal result
    // to bfloat16 BEFORE the first projection, matching the exact computation
    // path of the reference implementation.  Without this cast, proj_in
    // receives a float32 tensor and the autocast-internal cast may produce a
    // slightly different result, causing systematic emb differences (~2-4%)
    // that propagate into scale/shift errors (~0.13%) and ultimately
    // destabilise the attention computation across layers.
    auto emb =
        get_sinusoidal_embedding(timestep, proj_in_->options.in_features());
    // Cast to bfloat16 — mirrors official: sinusoidal(ts).to(bfloat16)
    emb = emb.to(torch::kBFloat16);
    emb = act_->forward(torch_linear(proj_in_, emb));
    emb = act_->forward(torch_linear(proj_hid_, emb));
    emb = torch_linear(proj_out_, emb);
    return emb;
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::Linear proj_hid_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::SiLU act_{nullptr};
};
TORCH_MODULE(ColaTimestepEmbedding);

// ---------------------------------------------------------------------------
// Rotary Embedding for ColaDiT (theta=10000, lang mode)
// ---------------------------------------------------------------------------

class ColaDiTRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  explicit ColaDiTRotaryEmbeddingImpl(int64_t dim) : dim_(dim) {
    // Compute inverse frequencies: inv_freq[i] = 1 / (10000^(2i/dim))
    auto inv_freq =
        1.0 /
        torch::pow(10000.0, torch::arange(0, dim, 2, torch::kFloat32) / dim);
    register_buffer("inv_freq", inv_freq);
  }

  // Compute cos/sin for positions [offset, offset+length).
  // Returns tensors of shape (length, dim).
  //
  // The official Cola-DLM TextRotaryEmbedding uses rotary_embedding_torch
  // which calls ``repeat(freqs, '... n -> ... (n r)', r=2)`` internally.
  // This REPEATS each frequency element twice in-place:
  //   [f0, f1, ..., f_{d/2-1}]  →  [f0, f0, f1, f1, ..., f_{d/2-1}, f_{d/2-1}]
  //
  // This is different from torch::cat({freqs, freqs}) which appends:
  //   [f0, f1, ..., f_{d/2-1}, f0, f1, ..., f_{d/2-1}]  (wrong!)
  //
  // With CONSECUTIVE-pair rotate_half (pairs (2k, 2k+1)):
  //   - repeat: both elements of each pair use the SAME angle f_k  ✓
  //   - cat:    elements 2k and 2k+1 use DIFFERENT angles f_k and f_{k+d/2}  ✗
  std::pair<torch::Tensor, torch::Tensor> get_cos_sin(int64_t length,
                                                      int64_t offset,
                                                      torch::Device device) {
    if (!inv_freq_.defined()) {
      inv_freq_ =
          1.0 / torch::pow(10000.0,
                           torch::arange(0, dim_, 2, torch::kFloat32) / dim_);
      inv_freq_ = inv_freq_.to(device);
    }
    auto positions = torch::arange(
        offset,
        offset + length,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto freqs = positions.unsqueeze(1) * inv_freq_.unsqueeze(0);
    // Repeat each frequency element twice to match rotary_embedding_torch's
    // ``repeat(freqs, '... n -> ... (n r)', r=2)``:
    //   [f0, f1, ..., f_{d/2-1}] → [f0, f0, f1, f1, ..., f_{d/2-1}, f_{d/2-1}]
    auto emb = freqs.repeat_interleave(2, /*dim=*/-1);
    return {torch::cos(emb), torch::sin(emb)};
  }

  // Apply rotary embedding to q and k tensors of shape (L, heads, head_dim).
  // cos/sin have shape (L, rope_dim) — DOUBLED frequencies from get_cos_sin.
  // Uses CONSECUTIVE-pair rotate_half matching rotary_embedding_torch:
  //   rotate_half[2k] = -x[2k+1], rotate_half[2k+1] = x[2k]
  // This means element 2k uses angle f_{2k} and element 2k+1 uses f_{2k+1}
  // (different angles per element), matching cola's TextRotaryEmbedding.
  // Rotates the first rope_dim dimensions; the rest pass through unchanged.
  static std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& cos,
      const torch::Tensor& sin) {
    auto apply_rope = [&](const torch::Tensor& t) {
      int64_t d = cos.size(-1);  // rope_dim (full, doubled)
      auto rot = t.narrow(-1, 0, d);
      auto pass = t.narrow(-1, d, t.size(-1) - d);

      // CONSECUTIVE rotate_half matching rotary_embedding_torch:
      // pairs (2k, 2k+1): result[2k]=-x[2k+1], result[2k+1]=x[2k]
      auto rotate_half = [](const torch::Tensor& x) {
        int64_t L_ = x.size(0), H_ = x.size(1), sz = x.size(-1);
        auto paired = x.reshape({L_, H_, sz / 2, 2});
        auto even = paired.select(-1, 0);  // x[0::2]
        auto odd = paired.select(-1, 1);   // x[1::2]
        return torch::stack({-odd, even}, /*dim=*/-1).reshape({L_, H_, sz});
      };

      auto c = cos.unsqueeze(1);  // (L, 1, d)
      auto s = sin.unsqueeze(1);

      auto rot_new = rot * c + rotate_half(rot) * s;
      return torch::cat({rot_new, pass}, /*dim=*/-1);
    };
    return {apply_rope(q), apply_rope(k)};
  }

  // Apply rotary embedding to a single tensor of shape (L, heads, head_dim).
  // cos/sin have shape (L, rope_dim) — DOUBLED. CONSECUTIVE rotate_half.
  static torch::Tensor apply_rotary_emb_single(const torch::Tensor& x,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin) {
    int64_t d = cos.size(-1);  // rope_dim (full, doubled)
    auto rot = x.narrow(-1, 0, d);
    auto pass = x.narrow(-1, d, x.size(-1) - d);

    // CONSECUTIVE rotate_half matching rotary_embedding_torch:
    // pairs (2k, 2k+1): result[2k]=-x[2k+1], result[2k+1]=x[2k]
    int64_t L_ = rot.size(0), H_ = rot.size(1), sz = rot.size(-1);
    auto paired = rot.reshape({L_, H_, sz / 2, 2});
    auto even = paired.select(-1, 0);  // x[0::2]
    auto odd = paired.select(-1, 1);   // x[1::2]
    auto rot_half =
        torch::stack({-odd, even}, /*dim=*/-1).reshape({L_, H_, sz});

    auto c = cos.unsqueeze(1);  // (L, 1, d)
    auto s = sin.unsqueeze(1);

    auto rot_new = rot * c + rot_half * s;
    return torch::cat({rot_new, pass}, /*dim=*/-1);
  }

 private:
  int64_t dim_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(ColaDiTRotaryEmbedding);

// ---------------------------------------------------------------------------
// Block-Causal Attention Mask (NA layout)
// ---------------------------------------------------------------------------

// Creates an additive block-causal attention mask for the NA (no-padding)
// flatten-concat layout. Returns (1, 1, L_q_total, L_k_total) with
// 0 at allowed positions and dtype::min elsewhere.
inline torch::Tensor create_block_causal_mask(
    const std::vector<int64_t>& k_lens,
    const std::vector<int64_t>& q_lens,
    int64_t block_size,
    torch::ScalarType dtype,
    torch::Device device) {
  int64_t B = k_lens.size();
  int64_t L_k = 0;
  int64_t L_q = 0;
  for (int64_t i = 0; i < B; ++i) {
    L_k += k_lens[i];
    L_q += q_lens[i];
  }

  // Compute cumulative sums
  std::vector<int64_t> k_cu(B + 1, 0);
  std::vector<int64_t> q_cu(B + 1, 0);
  for (int64_t i = 0; i < B; ++i) {
    k_cu[i + 1] = k_cu[i] + k_lens[i];
    q_cu[i + 1] = q_cu[i] + q_lens[i];
  }

  // Build index tensors
  auto k_sample = torch::zeros({L_k}, torch::kLong);
  auto k_local = torch::zeros({L_k}, torch::kLong);
  auto q_sample = torch::zeros({L_q}, torch::kLong);
  auto q_local = torch::zeros({L_q}, torch::kLong);

  for (int64_t b = 0; b < B; ++b) {
    int64_t k_len_b = k_lens[b];
    int64_t q_len_b = q_lens[b];
    if (k_len_b > 0) {
      k_sample.narrow(0, k_cu[b], k_len_b).fill_(b);
      // Use copy_() to in-place fill the view — plain = rebinds the local var.
      k_local.narrow(0, k_cu[b], k_len_b).copy_(torch::arange(k_len_b));
    }
    if (q_len_b > 0) {
      q_sample.narrow(0, q_cu[b], q_len_b).fill_(b);
      // Q refers to the LAST q_len_b positions of K within the same sample.
      q_local.narrow(0, q_cu[b], q_len_b)
          .copy_(torch::arange(k_len_b - q_len_b, k_len_b));
    }
  }

  // Compute block membership
  auto q_block = q_local.unsqueeze(1) / block_size;
  auto k_block = k_local.unsqueeze(0) / block_size;
  auto same_sample = q_sample.unsqueeze(1) == k_sample.unsqueeze(0);
  auto block_causal = q_block >= k_block;
  auto allowed = same_sample & block_causal;

  // Build additive mask
  float min_val = std::numeric_limits<float>::lowest();
  auto mask =
      torch::full({L_q, L_k},
                  min_val,
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  mask.masked_fill_(allowed.to(device), 0.0f);
  return mask.to(dtype).unsqueeze(0).unsqueeze(0);
}

// ---------------------------------------------------------------------------
// AdaLN (Adaptive Layer Normalization)
// ---------------------------------------------------------------------------

// AdaLN applies shift/scale modulation ("in" mode) and gate modulation
// ("out" mode) conditioned on the timestep embedding.
class AdaLNImpl : public torch::nn::Module {
 public:
  AdaLNImpl(int64_t dim,
            int64_t emb_dim,
            const std::vector<std::string>& layers,
            const std::vector<std::string>& modes = {"in", "out"})
      : layers_(layers), modes_(modes) {
    for (const auto& layer : layers) {
      if (std::find(modes.begin(), modes.end(), "in") != modes.end()) {
        // {layer}_in: SiLU -> Linear(dim, 2*dim) produces shift + scale
        auto seq = std::make_shared<torch::nn::SequentialImpl>();
        seq->push_back(torch::nn::SiLU());
        seq->push_back(torch::nn::Linear(dim, 2 * dim));
        register_module(layer + "_in", seq);
      }
      if (std::find(modes.begin(), modes.end(), "out") != modes.end()) {
        // {layer}_out: SiLU -> Linear(dim, dim) produces gate
        auto seq = std::make_shared<torch::nn::SequentialImpl>();
        seq->push_back(torch::nn::SiLU());
        seq->push_back(torch::nn::Linear(dim, dim));
        register_module(layer + "_out", seq);
      }
    }
  }

  // "in" mode: returns norm(hid) * (1 + scale) + shift
  // "out" mode: returns hid * gate + residual
  torch::Tensor forward(const torch::Tensor& hid,
                        const torch::Tensor& emb,
                        const std::string& layer,
                        const std::string& mode,
                        torch::nn::LayerNorm norm_layer = nullptr,
                        const torch::Tensor& residual = {}) {
    auto mod =
        named_modules()[layer + "_" + mode]->as<torch::nn::SequentialImpl>();
    // Manually expand Sequential(SiLU, Linear) to use torch::linear for
    // bit-identical matmul results with PyTorch's F.linear path.
    auto silu_out = torch::silu(emb);
    auto* lin = mod->named_modules()["1"]->as<torch::nn::LinearImpl>();
    torch::Tensor out;
    if (lin) {
      out = torch::linear(silu_out, lin->weight, lin->bias);
    } else {
      out = mod->forward(emb);  // fallback
    }

    // Repeat emb if it has fewer elements than hid (per-sample to per-token)
    if (out.size(0) != hid.size(0)) {
      // This shouldn't happen in the NA layout since emb is already per-token
      // but handle it for safety
      out = out.repeat_interleave(hid.size(0) / out.size(0), /*dim=*/0);
    }

    if (mode == "in") {
      auto chunks = out.chunk(2, /*dim=*/-1);
      auto shift = chunks[0];
      auto scale = chunks[1];
      return norm_layer(hid) * (1.0 + scale) + shift;
    }
    // "out" mode
    return hid * out + residual;
  }

 private:
  std::vector<std::string> layers_;
  std::vector<std::string> modes_;
};
TORCH_MODULE(AdaLN);

// ---------------------------------------------------------------------------
// MLP (GELU tanh approximation)
// ---------------------------------------------------------------------------

class ColaDiTMLPImpl : public torch::nn::Module {
 public:
  ColaDiTMLPImpl(int64_t dim, int64_t expand_ratio) {
    proj_in_ =
        register_module("proj_in", torch::nn::Linear(dim, dim * expand_ratio));
    act_ = register_module(
        "act", torch::nn::GELU(torch::nn::GELUOptions().approximate("tanh")));
    proj_out_ =
        register_module("proj_out", torch::nn::Linear(dim * expand_ratio, dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return torch_linear(proj_out_, act_->forward(torch_linear(proj_in_, x)));
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
};
TORCH_MODULE(ColaDiTMLP);

// ---------------------------------------------------------------------------
// ColaDiT Attention (QKV proj, QK norm, RoPE, block-causal attention)
// Implements per-sample KV cache matching the official Python ColaDiTAttention.
// ---------------------------------------------------------------------------

class ColaDiTAttentionImpl : public torch::nn::Module {
 public:
  ColaDiTAttentionImpl(int64_t txt_dim,
                       int64_t heads,
                       int64_t head_dim,
                       bool qk_bias,
                       int64_t rope_dim)
      : heads_(heads), head_dim_(head_dim) {
    int64_t inner_dim = heads * head_dim;
    proj_qkv_ = register_module(
        "proj_qkv",
        torch::nn::Linear(
            torch::nn::LinearOptions(txt_dim, inner_dim * 3).bias(qk_bias)));
    proj_out_ = register_module(
        "proj_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(inner_dim, txt_dim).bias(qk_bias)));
    norm_q_ = register_module(
        "norm_q",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    norm_k_ = register_module(
        "norm_k",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    rope_ = register_module("rope", ColaDiTRotaryEmbedding(rope_dim));
  }

  // Clear the per-sample KV cache.
  void set_kv_cache(bool /*flag*/) {
    k_cache_.clear();
    v_cache_.clear();
  }

  // Forward with optional KV cache support, matching the official Python
  // ColaDiTAttention.forward() semantics:
  //
  //   txt:          (L_q_total, txt_dim) — Q-side input (current block only)
  //   k_lens:       per-sample K-side lengths (cumulative: cache + current Q)
  //   q_lens:       per-sample Q-side lengths (= block_size during generation)
  //   update_kv:    append current K/V to cache, then read full cache as K
  //   use_kv_cache: prepend cached K/V to current K
  //
  // When both are False (unconditional pass): full_k = current K only.
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    int64_t L_q = txt.size(0);  // Q-side total length
    auto input_dtype = txt.dtype();
    int64_t B = static_cast<int64_t>(q_lens.size());

    // QKV projection — use torch::linear for bit-identical results with
    // PyTorch.
    auto qkv = torch_linear(proj_qkv_, txt);
    qkv = qkv.reshape({L_q, 3, heads_, head_dim_});
    auto txt_q = qkv.select(1, 0);  // (L_q, heads, head_dim)
    auto txt_k = qkv.select(1, 1);
    auto txt_v = qkv.select(1, 2);

    // QK normalization.
    txt_q = norm_q_(txt_q);
    txt_k = norm_k_(txt_k);

    // --- KV cache bookkeeping (matches official Python) ----------------------
    // Split per-sample new K/V from the Q-side projection.
    std::vector<torch::Tensor> new_ks, new_vs;
    {
      int64_t offset = 0;
      for (int64_t i = 0; i < B; ++i) {
        int64_t ql = q_lens[i];
        new_ks.push_back(txt_k.narrow(0, offset, ql));
        new_vs.push_back(txt_v.narrow(0, offset, ql));
        offset += ql;
      }
    }

    torch::Tensor full_k, full_v;
    if (update_kv) {
      // Append to cache, then read full cache.
      if (k_cache_.empty()) {
        k_cache_.resize(B);
        v_cache_.resize(B);
        for (int64_t i = 0; i < B; ++i) {
          k_cache_[i] = new_ks[i].clone();
          v_cache_[i] = new_vs[i].clone();
        }
      } else {
        for (int64_t i = 0; i < B; ++i) {
          k_cache_[i] = torch::cat({k_cache_[i], new_ks[i]}, /*dim=*/0);
          v_cache_[i] = torch::cat({v_cache_[i], new_vs[i]}, /*dim=*/0);
        }
      }
      std::vector<torch::Tensor> full_ks, full_vs;
      for (int64_t i = 0; i < B; ++i) {
        full_ks.push_back(k_cache_[i]);
        full_vs.push_back(v_cache_[i]);
      }
      full_k = torch::cat(full_ks, /*dim=*/0);
      full_v = torch::cat(full_vs, /*dim=*/0);
    } else if (use_kv_cache && !k_cache_.empty()) {
      // Prepend cached K/V to current K/V (don't update cache).
      std::vector<torch::Tensor> full_ks, full_vs;
      for (int64_t i = 0; i < B; ++i) {
        full_ks.push_back(torch::cat({k_cache_[i], new_ks[i]}, /*dim=*/0));
        full_vs.push_back(torch::cat({v_cache_[i], new_vs[i]}, /*dim=*/0));
      }
      full_k = torch::cat(full_ks, /*dim=*/0);
      full_v = torch::cat(full_vs, /*dim=*/0);
    } else {
      // No cache (unconditional pass): K = current Q only.
      full_k = txt_k;
      full_v = txt_v;
    }

    int64_t L_k_total = full_k.size(0);

    // --- RoPE ----------------------------------------------------------------
    // K positions: [0, k_lens[i]) for each sample i.
    // Q positions: [k_lens[i] - q_lens[i], k_lens[i]) — tail of K.
    torch::Tensor cos_k, sin_k, cos_q, sin_q;
    {
      std::vector<torch::Tensor> ck_list, sk_list, cq_list, sq_list;
      for (int64_t i = 0; i < B; ++i) {
        auto [ck, sk] = rope_->get_cos_sin(k_lens[i], 0, txt.device());
        ck_list.push_back(ck);
        sk_list.push_back(sk);
        int64_t q_offset = k_lens[i] - q_lens[i];
        auto [cq, sq] = rope_->get_cos_sin(q_lens[i], q_offset, txt.device());
        cq_list.push_back(cq);
        sq_list.push_back(sq);
      }
      cos_k = torch::cat(ck_list, /*dim=*/0);
      sin_k = torch::cat(sk_list, /*dim=*/0);
      cos_q = torch::cat(cq_list, /*dim=*/0);
      sin_q = torch::cat(sq_list, /*dim=*/0);
    }

    // Apply RoPE in float32 to match official Python:
    //   apply_rotary_emb(freqs, txt_q.float()).to(txt_q.dtype)
    // Without this, bfloat16 cos/sin introduces per-dimension errors that
    // accumulate across layers and cause feature-direction drift.
    {
      auto q_dtype = txt_q.dtype();
      txt_q = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
                  txt_q.to(torch::kFloat32),
                  cos_q.to(torch::kFloat32),
                  sin_q.to(torch::kFloat32))
                  .to(q_dtype);
    }
    {
      auto k_dtype = full_k.dtype();
      full_k = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
                   full_k.to(torch::kFloat32),
                   cos_k.to(torch::kFloat32),
                   sin_k.to(torch::kFloat32))
                   .to(k_dtype);
    }

    // --- Attention -----------------------------------------------------------
    // Mirror official slow_attn: Q/K/V in bfloat16, softmax under bf16 autocast
    // so CUDA promotes softmax to fp32 internally (matching training numerics).
    auto q_na = txt_q.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);
    auto k_na = full_k.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);
    auto v_na = full_v.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);

    torch::Tensor attn_out;
    {
      torch::NoGradGuard no_grad;
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
      auto attn = q_na.mul(scale).matmul(k_na.transpose(-2, -1));
      if (attn_mask.defined()) {
        attn = attn + attn_mask.to(torch::kBFloat16);
      }
      auto attn_weight = torch::softmax(attn, /*dim=*/-1);
      attn_out = attn_weight.matmul(v_na);
    }

    attn_out = attn_out.squeeze(0).permute({1, 0, 2}).reshape(
        {L_q, heads_ * head_dim_});
    return torch_linear(proj_out_, attn_out.to(input_dtype));  // (L_q, txt_dim)
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  torch::nn::Linear proj_qkv_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::LayerNorm norm_q_{nullptr};
  torch::nn::LayerNorm norm_k_{nullptr};
  ColaDiTRotaryEmbedding rope_{nullptr};
  // Per-sample KV cache: one (l_i_cum, heads, head_dim) tensor per sample.
  std::vector<torch::Tensor> k_cache_;
  std::vector<torch::Tensor> v_cache_;
};
TORCH_MODULE(ColaDiTAttention);

// ---------------------------------------------------------------------------
// ColaDiT Transformer Block
// ---------------------------------------------------------------------------

class ColaDiTBlockImpl : public torch::nn::Module {
 public:
  ColaDiTBlockImpl(int64_t txt_dim,
                   int64_t emb_dim,
                   int64_t heads,
                   int64_t head_dim,
                   int64_t expand_ratio,
                   float norm_eps,
                   bool qk_bias,
                   int64_t rope_dim) {
    msa_norm_ = register_module(
        "msa_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({txt_dim})
                                 .eps(norm_eps)
                                 .elementwise_affine(false)));
    msa_ = register_module(
        "msa", ColaDiTAttention(txt_dim, heads, head_dim, qk_bias, rope_dim));
    mlp_norm_ = register_module(
        "mlp_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({txt_dim})
                                 .eps(norm_eps)
                                 .elementwise_affine(false)));
    mlp_ = register_module("mlp", ColaDiTMLP(txt_dim, expand_ratio));
    ada_ = register_module(
        "ada", AdaLN(txt_dim, emb_dim, std::vector<std::string>{"msa", "mlp"}));
  }

  // txt:          (L_q, dim) — Q-side hidden state (current block only)
  // k_lens:       per-sample cumulative K lengths (cache + current Q)
  // q_lens:       per-sample Q lengths (= block_size during generation)
  // emb:          (L_q, emb_dim) — AdaLN conditioning for Q positions only
  // update_kv / use_kv_cache: forwarded to ColaDiTAttention KV cache logic
  // Diagnostic: block index, set by ColaDiTTransformerImpl after construction.
  int64_t block_idx_ = -1;
  // Only log on the first forward() call per inference pass (reset by
  // set_kv_cache).
  mutable int64_t blk_fwd_count_ = 0;

  void set_kv_cache(bool flag) {
    msa_->set_kv_cache(flag);
    blk_fwd_count_ = 0;
  }

  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& emb,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    // Attention sublayer: AdaLN (in) → attention → AdaLN (out) + residual
    // All tensors are float32 (autocast disabled by caller
    // ColaDiTTransformerImpl). This eliminates cross-framework bfloat16 GEMM
    // differences.
    auto txt_msa = ada_->forward(txt, emb, "msa", "in", msa_norm_);

    // Per-step diagnostics for blk[0] and blk[1].
    // blk[0]: track dim7 to match official cola-log
    // (after_adaln_in/txt_in/final_out dim7). blk[1]: track dim1 to trace where
    // net contribution sign flips.
    const bool do_blk1_trace = (block_idx_ == 1 && blk_fwd_count_ == 0);
    const bool do_blk0_trace = (block_idx_ == 0 && blk_fwd_count_ == 0);
    // Log both dim1 and dim7 for blk[0] so we can compare with official dim7
    // data.
    auto _logdim = [this](const torch::Tensor& t,
                          const char* label,
                          int64_t dim_idx = 1) {
      int64_t gen_pos = 11;
      if (t.size(0) > gen_pos && t.size(-1) > dim_idx) {
        auto f = t[gen_pos].cpu().to(torch::kFloat32);
        LOG(INFO) << "trace " << label << " gen0_dim" << dim_idx << "="
                  << f[dim_idx].item<float>();
        // Also always log dim7 for blk[0] (matches official cola-log trace
        // format)
        if (block_idx_ == 0 && t.size(-1) > 7) {
          LOG(INFO) << "trace " << label << " gen0_dim7=" << f[7].item<float>();
        }
      }
    };
    if (do_blk1_trace) {
      _logdim(txt, "blk1_txt_in");         // residual (blk[0] output)
      _logdim(txt_msa, "blk1_adaln_msa");  // after AdaLN_in for msa
    }
    if (do_blk0_trace) {
      // Log txt_in (PatchIn1D output) with dim7 — matches official
      // "txt_in_residual"
      _logdim(txt, "blk0_txt_in");
      _logdim(txt_msa, "blk0_adaln_msa");  // matches official "after_adaln_in"
    }

    // Diagnostic for block 8 (first forward call only): mirrors cola-log.txt
    // Also increment counter for blk[0] and blk[1] trace paths so they only
    // fire once per inference pass (same semantics as blk[8]).
    const bool do_blk8_log = (block_idx_ == 8 && blk_fwd_count_++ == 0);
    if (do_blk0_trace || do_blk1_trace) {
      blk_fwd_count_++;
    }
    if (do_blk8_log) {
      auto _s = [](const torch::Tensor& t, const char* label) {
        auto f = t.cpu().to(torch::kFloat32);
        LOG(INFO) << "DiT-blk[8]: " << label
                  << " mean=" << f.mean().item<float>()
                  << " std=" << f.std().item<float>()
                  << " max=" << f.abs().max().item<float>();
      };
      _s(txt_msa, "AdaLN_in(txt) → attn input");
      // ada.msa_in(emb) → [shift, scale] — key diagnostic
      {
        auto* seq =
            ada_->named_modules()["msa_in"]->as<torch::nn::SequentialImpl>();
        if (seq) {
          auto msa_in_out = seq->forward(emb.to(torch::kFloat32));
          auto chunks = msa_in_out.chunk(2, /*dim=*/-1);
          _s(chunks[0], "ada.msa_in shift");
          _s(chunks[1], "ada.msa_in scale");
          // Print first 8 values of scale for gen-position token 11
          // to directly compare with official cola-log.txt
          {
            auto sc_cpu = chunks[1].cpu();  // (16, 2048) float32
            std::string vals;
            for (int64_t d = 0; d < std::min(int64_t(8), sc_cpu.size(-1));
                 ++d) {
              vals += std::to_string(sc_cpu[11][d].item<float>()) + " ";
            }
            LOG(INFO) << "DiT-blk[8]: ada.msa_in scale[gen0] first 8: [" << vals
                      << "]";
          }
          // Print first 8 values of LayerNorm(txt) for gen-position token 11
          auto norm_txt = msa_norm_(txt.to(torch::kFloat32));
          _s(norm_txt, "LayerNorm(txt)");
          {
            auto n_cpu = norm_txt.cpu();
            std::string vals;
            for (int64_t d = 0; d < std::min(int64_t(8), n_cpu.size(-1)); ++d) {
              vals += std::to_string(n_cpu[11][d].item<float>()) + " ";
            }
            LOG(INFO) << "DiT-blk[8]: LayerNorm(txt)[gen0] first 8: [" << vals
                      << "]";
          }
        }
      }
      // gate = ada.msa_out(emb) and ada.mlp_out(emb)
      // Use the same named_modules()[key] access pattern as AdaLNImpl::forward.
      {
        auto* seq =
            ada_->named_modules()["msa_out"]->as<torch::nn::SequentialImpl>();
        if (seq) {
          _s(seq->forward(emb.to(torch::kFloat32)), "ada.msa_out gate");
        }
      }
      {
        auto* seq =
            ada_->named_modules()["mlp_out"]->as<torch::nn::SequentialImpl>();
        if (seq) {
          _s(seq->forward(emb.to(torch::kFloat32)), "ada.mlp_out gate");
        }
      }
    }

    txt_msa = msa_->forward(txt_msa,
                            k_lens,
                            q_lens,
                            block_size,
                            attn_mask,
                            update_kv,
                            use_kv_cache);
    if (do_blk1_trace) {
      _logdim(txt_msa, "blk1_attn_out");
    }
    if (do_blk0_trace) {
      _logdim(txt_msa, "blk0_attn_out");
    }

    if (do_blk8_log) {
      auto f = txt_msa.cpu().to(torch::kFloat32);
      LOG(INFO) << "DiT-blk[8]: attn_out mean=" << f.mean().item<float>()
                << " std=" << f.std().item<float>()
                << " max=" << f.abs().max().item<float>();
    }

    auto txt_out = ada_->forward(txt_msa, emb, "msa", "out", nullptr, txt);
    if (do_blk1_trace) {
      _logdim(txt_out, "blk1_after_msa_out");
    }
    if (do_blk0_trace) {
      _logdim(txt_out, "blk0_after_msa_out");
    }

    if (do_blk8_log) {
      auto f = txt_out.cpu().to(torch::kFloat32);
      LOG(INFO) << "DiT-blk[8]: after_msa_out mean=" << f.mean().item<float>()
                << " std=" << f.std().item<float>()
                << " max=" << f.abs().max().item<float>();
    }

    // MLP sublayer: AdaLN (in) → MLP → AdaLN (out) + residual
    auto txt_mlp = ada_->forward(txt_out, emb, "mlp", "in", mlp_norm_);
    if (do_blk1_trace) {
      _logdim(txt_mlp, "blk1_adaln_mlp");
    }
    if (do_blk0_trace) {
      _logdim(txt_mlp, "blk0_adaln_mlp");
    }

    if (do_blk8_log) {
      auto f = txt_mlp.cpu().to(torch::kFloat32);
      LOG(INFO) << "DiT-blk[8]: mlp_in (AdaLN_in mlp) mean="
                << f.mean().item<float>() << " std=" << f.std().item<float>()
                << " max=" << f.abs().max().item<float>();
    }

    txt_mlp = mlp_->forward(txt_mlp);
    if (do_blk1_trace) {
      _logdim(txt_mlp, "blk1_mlp_out");
    }
    if (do_blk0_trace) {
      _logdim(txt_mlp, "blk0_mlp_out");
    }

    if (do_blk8_log) {
      auto f = txt_mlp.cpu().to(torch::kFloat32);
      LOG(INFO) << "DiT-blk[8]: mlp_out mean=" << f.mean().item<float>()
                << " std=" << f.std().item<float>()
                << " max=" << f.abs().max().item<float>();
    }

    txt_out = ada_->forward(txt_mlp, emb, "mlp", "out", nullptr, txt_out);
    if (do_blk1_trace) {
      _logdim(txt_out, "blk1_final_out");
    }
    if (do_blk0_trace) {
      _logdim(txt_out, "blk0_final_out");
    }
    return txt_out;
  }

 private:
  torch::nn::LayerNorm msa_norm_{nullptr};
  ColaDiTAttention msa_{nullptr};
  torch::nn::LayerNorm mlp_norm_{nullptr};
  ColaDiTMLP mlp_{nullptr};
  AdaLN ada_{nullptr};
};
TORCH_MODULE(ColaDiTBlock);

// ---------------------------------------------------------------------------
// PatchIn1D / PatchOut1D — patchification wrappers
// Matches the Python PatchIn1D/PatchOut1D classes which contain a Linear "proj"
// ---------------------------------------------------------------------------

class PatchIn1DImpl : public torch::nn::Module {
 public:
  PatchIn1DImpl(int64_t in_channels, int64_t patch_size, int64_t dim) {
    proj_ = register_module("proj",
                            torch::nn::Linear(in_channels * patch_size, dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return torch_linear(proj_, x);
  }

 private:
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(PatchIn1D);

class PatchOut1DImpl : public torch::nn::Module {
 public:
  PatchOut1DImpl(int64_t out_channels, int64_t patch_size, int64_t dim) {
    proj_ = register_module("proj",
                            torch::nn::Linear(dim, out_channels * patch_size));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return torch_linear(proj_, x);
  }

 private:
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(PatchOut1D);

// ---------------------------------------------------------------------------
// ColaDiT Transformer (full model)
// ---------------------------------------------------------------------------

class ColaDiTTransformerImpl : public torch::nn::Module {
 public:
  explicit ColaDiTTransformerImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t txt_dim = args.txt_dim();
    int64_t emb_dim = args.emb_dim();
    int64_t heads = args.heads();
    int64_t head_dim = args.head_dim();
    int64_t expand_ratio = args.expand_ratio();
    int64_t num_layers = args.num_layers();
    float norm_eps = args.norm_eps();
    bool qk_bias = args.qk_bias();
    int64_t rope_dim = args.rope_dim();
    int64_t txt_in_channels = args.txt_in_channels();
    int64_t txt_out_channels = args.txt_out_channels();
    block_size_ = args.block_size();
    txt_dim_ = txt_dim;

    // Input projection: latent_dim -> txt_dim (PatchIn1D wraps Linear as
    // "proj")
    txt_in_ = register_module("txt_in", PatchIn1D(txt_in_channels, 1, txt_dim));

    // Timestep embedding: sinusoidal(256) -> MLP -> emb_dim
    emb_in_ =
        register_module("emb_in", ColaTimestepEmbedding(256, txt_dim, emb_dim));

    // Transformer blocks
    blocks_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = ColaDiTBlock(txt_dim,
                                emb_dim,
                                heads,
                                head_dim,
                                expand_ratio,
                                norm_eps,
                                qk_bias,
                                rope_dim);
      block->block_idx_ = i;  // set diagnostic index
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    // Output: norm + AdaLN + projection
    txt_out_norm_ = register_module(
        "txt_out_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({txt_dim}).eps(norm_eps)));
    txt_out_ada_ = register_module("txt_out_ada",
                                   AdaLN(txt_dim,
                                         emb_dim,
                                         std::vector<std::string>{"out"},
                                         std::vector<std::string>{"in"}));
    txt_out_ =
        register_module("txt_out", PatchOut1D(txt_out_channels, 1, txt_dim));
  }

  // Clear per-layer KV caches on all blocks; reset diagnostic counter so the
  // first forward of each new inference call is always logged.
  void set_kv_cache(bool flag) {
    for (auto& block : blocks_) {
      block->set_kv_cache(flag);
    }
    fwd_log_counter_ = 0;
  }

  // Forward pass — matches the official Python ColaDiTModel.forward().
  //
  // txt:          (L_q_total, txt_in_channels) — Q-side input (current block
  //               or prefix, depending on update_kv flag)
  // k_lens:       per-sample cumulative K lengths (cache + current Q)
  // q_lens:       per-sample Q lengths (= block_size during generation, or
  //               prefix length during prefetch with update_kv=True)
  // timestep:     per-token timestep (length = L_q_total)
  // update_kv:    commit current Q's K/V to per-layer cache
  // use_kv_cache: read per-layer cached K/V for the conditional pass
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& timestep,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    int64_t L_q = txt.size(0);

    // Project Q-side input only.
    auto hidden = txt_in_(txt);  // (L_q, txt_dim)

    // Timestep embedding — length = L_q.
    auto ts = timestep;
    if (ts.dim() == 0) {
      ts = ts.unsqueeze(0);
    }
    if (ts.size(0) == 1 && L_q > 1) {
      ts = ts.expand({L_q}, /*implicit=*/true);
    }
    // Pass timestep as-is; ColaTimestepEmbeddingImpl::forward will compute the
    // sinusoidal embedding in float32 and then cast to bfloat16 internally,
    // matching the official Python path exactly.
    auto emb = emb_in_(ts);  // (L_q, emb_dim)

    // Block-causal attention mask for Q × K.
    auto attn_mask = create_block_causal_mask(
        k_lens, q_lens, block_size_, hidden.scalar_type(), hidden.device());

    // Diagnostic: log per-block hidden state on first forward call only,
    // to compare with official cola-log DiT-blk[bi] txt_in/txt_out entries.
    // Controlled by a call counter; reset by set_kv_cache(false).
    bool do_log_blocks = (fwd_log_counter_++ == 0);
    if (do_log_blocks) {
      auto ts_f = ts.cpu().to(torch::kFloat32);
      auto emb_f = emb.cpu().to(torch::kFloat32);
      LOG(INFO) << "DiT: ts mean=" << ts_f.mean().item<float>()
                << ", min=" << ts_f.min().item<float>()
                << ", max=" << ts_f.max().item<float>()
                << " | emb mean=" << emb_f.mean().item<float>()
                << ", std=" << emb_f.std().item<float>();

      // Log emb stats separately for prompt positions (t=0) and gen positions
      // (t=1000) to identify whether the emb difference causes blk[8] behavior.
      int64_t L_emb = emb_f.size(0);
      if (L_emb > 1) {
        // Find first gen position: scan ts for the first non-zero value
        int64_t first_gen = L_emb;
        for (int64_t i = 0; i < L_emb; ++i) {
          if (ts_f[i].item<float>() > 0.0f) {
            first_gen = i;
            break;
          }
        }
        if (first_gen > 0 && first_gen < L_emb) {
          auto emb_prompt = emb_f.narrow(0, 0, first_gen);
          auto emb_gen = emb_f.narrow(0, first_gen, L_emb - first_gen);
          LOG(INFO) << "DiT: emb_prompt(t=0, pos 0:" << first_gen
                    << ") mean=" << emb_prompt.mean().item<float>()
                    << " std=" << emb_prompt.std().item<float>()
                    << " max=" << emb_prompt.abs().max().item<float>()
                    << " | emb_gen(t=1000, pos " << first_gen << ":" << L_emb
                    << ") mean=" << emb_gen.mean().item<float>()
                    << " std=" << emb_gen.std().item<float>()
                    << " max=" << emb_gen.abs().max().item<float>();
          // Log first few values of emb_gen for direct comparison with official
          auto eg0 = emb_gen[0].cpu();  // first gen position
          std::string eg0_vals;
          for (int64_t di = 0; di < std::min(int64_t(8), eg0.size(0)); ++di) {
            eg0_vals += std::to_string(eg0[di].item<float>()) + " ";
          }
          LOG(INFO) << "DiT: emb_gen[0] first 8 dims: [" << eg0_vals << "]";
        }
      }
    }
    auto _log_hidden =
        [&do_log_blocks](const torch::Tensor& h, int64_t bi, const char* tag) {
          if (!do_log_blocks) {
            return;
          }
          auto hf = h.cpu().to(torch::kFloat32);
          LOG(INFO) << "DiT-blk[" << bi << "]: " << tag
                    << ": mean=" << hf.mean().item<float>()
                    << ", std=" << hf.std().item<float>();
        };

    for (int64_t bi = 0; bi < static_cast<int64_t>(blocks_.size()); ++bi) {
      _log_hidden(hidden, bi, "txt_in");

      hidden = blocks_[bi]->forward(hidden,
                                    k_lens,
                                    q_lens,
                                    emb,
                                    block_size_,
                                    attn_mask,
                                    update_kv,
                                    use_kv_cache);
      _log_hidden(hidden, bi, "txt_out");

      // Log first 8 values at gen-position token (pos 11) for all blk[0..8]
      // to trace per-dimension divergence between xllm and official.
      if (do_log_blocks && bi <= 8) {
        int64_t gen_pos = 11;  // first gen token position
        if (hidden.size(0) > gen_pos) {
          auto h_cpu = hidden[gen_pos].cpu().to(torch::kFloat32);
          std::string vals;
          for (int64_t d = 0; d < std::min(int64_t(8), h_cpu.size(0)); ++d) {
            vals += std::to_string(h_cpu[d].item<float>()) + " ";
          }
          LOG(INFO) << "DiT-blk[" << bi << "]: txt_out[gen0] first 8: [" << vals
                    << "]";
        }
      }
    }

    // Output: AdaLN + projection.
    hidden = txt_out_ada_->forward(hidden, emb, "out", "in", txt_out_norm_);
    hidden = txt_out_(hidden);

    return hidden;  // (L_q, txt_out_channels)
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_module_from_state_dicts(*loader, this);

    // Diagnostic: log emb_in_ weight norms to verify TimestepEmbedding loading.
    {
      auto emb_params = emb_in_->named_parameters(/*recurse=*/true);
      for (auto& kv : emb_params) {
        auto w = kv.value().to(torch::kFloat32);
        int64_t s0 = w.size(0);
        int64_t s1 = (w.dim() > 1) ? w.size(1) : 1;
        LOG(INFO) << "Cola-DIT emb_in " << kv.key()
                  << " norm=" << w.norm().item<float>()
                  << " mean=" << w.mean().item<float>() << " shape=[" << s0
                  << "," << s1 << "]";
      }
    }

    // Diagnostic: log key weight norms for blk[0] and blk[8] to verify
    // that checkpoint loading is correct. blk[8] weights should differ from
    // blk[0] weights if loading is correct.
    for (int64_t bi : {0, 8}) {
      if (bi >= static_cast<int64_t>(blocks_.size())) {
        continue;
      }
      auto named_params = blocks_[bi]->named_parameters(/*recurse=*/true);
      for (auto& kv : named_params) {
        const auto& key = kv.key();
        // Log ada weights+biases and proj_qkv weight
        bool is_ada = key.find("ada") != std::string::npos;
        bool is_qkv = key.find("proj_qkv") != std::string::npos &&
                      key.find("weight") != std::string::npos;
        bool is_wt = key.find("weight") != std::string::npos;
        bool is_bias = key.find("bias") != std::string::npos;
        if (is_qkv || (is_ada && (is_wt || is_bias))) {
          auto w = kv.value().to(torch::kFloat32);
          int64_t s0 = w.size(0);
          int64_t s1 = (w.dim() > 1) ? w.size(1) : 1;
          LOG(INFO) << "Cola-DIT blk[" << bi << "] " << key
                    << " norm=" << w.norm().item<float>()
                    << " mean=" << w.mean().item<float>() << " shape=[" << s0
                    << "," << s1 << "]";
        }
      }
    }
  }

  int64_t block_size() const { return block_size_; }

 private:
  int64_t block_size_;
  int64_t txt_dim_;
  PatchIn1D txt_in_{nullptr};
  ColaTimestepEmbedding emb_in_{nullptr};
  std::vector<ColaDiTBlock> blocks_;
  torch::nn::LayerNorm txt_out_norm_{nullptr};
  AdaLN txt_out_ada_{nullptr};
  PatchOut1D txt_out_{nullptr};
  // Diagnostic: counter of forward() calls since last set_kv_cache().
  // The first call per inference pass logs per-block hidden states.
  int64_t fwd_log_counter_ = 0;
};
TORCH_MODULE(ColaDiTTransformer);

// ---------------------------------------------------------------------------
// REGISTER_MODEL_ARGS for ColaDiT config.json
// ---------------------------------------------------------------------------

REGISTER_MODEL_ARGS(cola_dit, [&] {
  LOAD_ARG_OR(model_type, "model_type", "cola_dit");
  LOAD_ARG(txt_dim, "txt_dim");
  LOAD_ARG(txt_in_channels, "txt_in_channels");
  LOAD_ARG(txt_out_channels, "txt_out_channels");
  LOAD_ARG(emb_dim, "emb_dim");
  LOAD_ARG(heads, "heads");
  LOAD_ARG(head_dim, "head_dim");
  LOAD_ARG(expand_ratio, "expand_ratio");
  LOAD_ARG(num_layers, "num_layers");
  LOAD_ARG(rope_dim, "rope_dim");
  LOAD_ARG(block_size, "block_size");
  LOAD_ARG(qk_bias, "qk_bias");
  LOAD_ARG_OR(norm_eps, "norm_eps", 1e-5);
  LOAD_ARG_OR(patch_size, "patch_size", 1);
});

}  // namespace xllm
