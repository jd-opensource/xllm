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
  auto exponent = -std::log(10000.0f) *
                  torch::arange(0,
                                half_dim,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(timesteps.device())) /
                  static_cast<float>(half_dim);
  auto emb = timesteps.to(torch::kFloat32).unsqueeze(1) * exponent.unsqueeze(0);
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
    // Cast to model dtype (bf16) to match Python's autocast behavior.
    auto model_dtype = proj_in_->weight.dtype();
    auto emb =
        get_sinusoidal_embedding(timestep, proj_in_->options.in_features())
            .to(model_dtype);
    emb = act_->forward(proj_in_->forward(emb));
    emb = act_->forward(proj_hid_->forward(emb));
    emb = proj_out_->forward(emb);
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

  // Compute cos/sin for positions [offset, offset+length)
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
    auto cos = torch::cos(freqs);
    auto sin = torch::sin(freqs);
    return {cos, sin};
  }

  // Apply rotary embedding to q and k tensors of shape (L, heads, head_dim)
  // cos/sin have shape (L, dim/2)
  static std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& cos,
      const torch::Tensor& sin) {
    // q, k: (L, heads, head_dim)
    // cos, sin: (L, rope_dim) -- only applied to first rope_dim dimensions
    int64_t d = cos.size(-1);
    auto q_rot = q.narrow(-1, 0, d);
    auto q_pass = q.narrow(-1, d, q.size(-1) - d);
    auto k_rot = k.narrow(-1, 0, d);
    auto k_pass = k.narrow(-1, d, k.size(-1) - d);

    // rotate_half: [-x2, x1] for each pair
    auto rotate_half = [](const torch::Tensor& x) {
      auto d = x.size(-1);
      return torch::cat({-x.narrow(-1, d / 2, d / 2), x.narrow(-1, 0, d / 2)},
                        /*dim=*/-1);
    };

    auto cos_expanded = cos.unsqueeze(1);  // (L, 1, d)
    auto sin_expanded = sin.unsqueeze(1);

    auto q_new = torch::cat(
        {q_rot * cos_expanded + rotate_half(q_rot) * sin_expanded, q_pass},
        /*dim=*/-1);
    auto k_new = torch::cat(
        {k_rot * cos_expanded + rotate_half(k_rot) * sin_expanded, k_pass},
        /*dim=*/-1);
    return {q_new, k_new};
  }

  // Apply rotary embedding to a single tensor of shape (L, heads, head_dim)
  static torch::Tensor apply_rotary_emb_single(const torch::Tensor& x,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin) {
    int64_t d = cos.size(-1);
    auto x_rot = x.narrow(-1, 0, d);
    auto x_pass = x.narrow(-1, d, x.size(-1) - d);

    auto rotate_half = [](const torch::Tensor& t) {
      auto sz = t.size(-1);
      return torch::cat(
          {-t.narrow(-1, sz / 2, sz / 2), t.narrow(-1, 0, sz / 2)},
          /*dim=*/-1);
    };

    auto cos_expanded = cos.unsqueeze(1);
    auto sin_expanded = sin.unsqueeze(1);

    return torch::cat(
        {x_rot * cos_expanded + rotate_half(x_rot) * sin_expanded, x_pass},
        /*dim=*/-1);
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
      k_local.narrow(0, k_cu[b], k_len_b) = torch::arange(k_len_b);
    }
    if (q_len_b > 0) {
      q_sample.narrow(0, q_cu[b], q_len_b).fill_(b);
      // Q refers to the LAST q_len_b positions of K within the same sample
      q_local.narrow(0, q_cu[b], q_len_b) =
          torch::arange(k_len_b - q_len_b, k_len_b);
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
    auto out = mod->forward(emb);

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
    return proj_out_->forward(act_->forward(proj_in_->forward(x)));
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
};
TORCH_MODULE(ColaDiTMLP);

// ---------------------------------------------------------------------------
// ColaDiT Attention (QKV proj, QK norm, RoPE, block-causal attention)
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

  // Returns (K_len, heads * head_dim). Only the last Q_len rows contain
  // meaningful attention output; the prefix rows are passed through unchanged
  // so that residual connections in the block can operate on K_len uniformly.
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        int64_t block_size,
                        const torch::Tensor& attn_mask) {
    int64_t L_k = txt.size(0);  // full K-side sequence length
    auto input_dtype = txt.dtype();
    int64_t B = k_lens.size();

    // Compute total Q_len
    int64_t L_q = 0;
    for (auto ql : q_lens) {
      L_q += ql;
    }

    // QKV projection over full K-side sequence
    auto qkv = proj_qkv_(txt);
    qkv = qkv.reshape({L_k, 3, heads_, head_dim_});
    auto txt_q = qkv.select(1, 0);  // (L_k, heads, head_dim)
    auto txt_k = qkv.select(1, 1);
    auto txt_v = qkv.select(1, 2);

    // QK normalization
    txt_q = norm_q_(txt_q);
    txt_k = norm_k_(txt_k);

    // Apply RoPE to full K_len sequence.
    // Positions [0, K_len) for all rows. Q uses the last L_q rows, which
    // corresponds to positions [K_len-L_q, K_len) — matching the Python
    // reference where q_offset = txt_shape - txt_q_shape.
    torch::Tensor cos_all, sin_all;
    {
      std::vector<torch::Tensor> cos_list, sin_list;
      for (int64_t i = 0; i < B; ++i) {
        auto [cos, sin] = rope_->get_cos_sin(k_lens[i], 0, txt.device());
        cos_list.push_back(cos);
        sin_list.push_back(sin);
      }
      cos_all = torch::cat(cos_list, /*dim=*/0);
      sin_all = torch::cat(sin_list, /*dim=*/0);
    }

    txt_q = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
        txt_q, cos_all, sin_all);
    txt_k = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
        txt_k, cos_all, sin_all);

    // Extract Q as last L_q rows for attention
    auto q_for_attn =
        txt_q.narrow(0, L_k - L_q, L_q);  // (L_q, heads, head_dim)

    // Reshape for attention: Q=(1, heads, L_q, head_dim), K/V=(1, heads, L_k,
    // head_dim)
    auto q_na = q_for_attn.permute({1, 0, 2}).unsqueeze(0);
    auto k_na = txt_k.permute({1, 0, 2}).unsqueeze(0);
    auto v_na = txt_v.permute({1, 0, 2}).unsqueeze(0);

    // Compute attention with mask
    torch::Tensor attn_out;
    {
      torch::NoGradGuard no_grad;
      float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
      auto attn = q_na.to(torch::kBFloat16)
                      .mul(scale)
                      .matmul(k_na.to(torch::kBFloat16).transpose(-2, -1));
      if (attn_mask.defined()) {
        attn = attn + attn_mask.to(attn.dtype());
      }
      auto attn_weight = torch::softmax(attn, /*dim=*/-1);
      attn_out = attn_weight.matmul(v_na.to(torch::kBFloat16));
    }

    // Reshape attention output to (L_q, heads * head_dim)
    attn_out = attn_out.squeeze(0).permute({1, 0, 2}).reshape(
        {L_q, heads_ * head_dim_});
    auto q_out = proj_out_(attn_out.to(input_dtype));  // (L_q, txt_dim)

    // Place Q output into a K_len tensor (prefix rows unchanged)
    torch::Tensor out = txt.clone();
    out.narrow(0, L_k - L_q, L_q).copy_(q_out);
    return out;
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  torch::nn::Linear proj_qkv_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::LayerNorm norm_q_{nullptr};
  torch::nn::LayerNorm norm_k_{nullptr};
  ColaDiTRotaryEmbedding rope_{nullptr};
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

  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& emb,
                        int64_t block_size,
                        const torch::Tensor& attn_mask) {
    // Attention sublayer with AdaLN
    auto txt_msa = ada_->forward(txt, emb, "msa", "in", msa_norm_);
    txt_msa = msa_->forward(txt_msa, k_lens, q_lens, block_size, attn_mask);
    auto txt_out = ada_->forward(txt_msa, emb, "msa", "out", nullptr, txt);

    // MLP sublayer with AdaLN
    auto txt_mlp = ada_->forward(txt_out, emb, "mlp", "in", mlp_norm_);
    txt_mlp = mlp_->forward(txt_mlp);
    txt_out = ada_->forward(txt_mlp, emb, "mlp", "out", nullptr, txt_out);
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

  torch::Tensor forward(const torch::Tensor& x) { return proj_(x); }

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

  torch::Tensor forward(const torch::Tensor& x) { return proj_(x); }

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

  // Forward pass
  // txt: (L_q_total, txt_in_channels) - current noisy latent block
  // k_lens: per-sample K-side lengths
  // q_lens: per-sample Q-side lengths (block_size during generation)
  // timestep: flow matching time t
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& timestep) {
    // Cast to model dtype (bf16) to match Python's autocast behavior.
    auto model_dtype = parameters().front().dtype();
    auto hidden = txt_in_(txt.to(model_dtype));

    // Compute timestep embedding for full K-side sequence.
    // Blocks keep hidden as K_len throughout; AdaLN "in" operates on full seq.
    auto ts = timestep;
    if (ts.dim() == 0) {
      ts = ts.unsqueeze(0);
    }
    if (ts.size(0) != hidden.size(0)) {
      if (ts.size(0) == 1) {
        ts = ts.expand({hidden.size(0)}, /*implicit=*/true);
      } else {
        int64_t repeat_factor = (hidden.size(0) + ts.size(0) - 1) / ts.size(0);
        ts = ts.repeat({repeat_factor}).narrow(0, 0, hidden.size(0));
      }
    }
    auto emb = emb_in_(ts);  // (K_len, emb_dim)

    // Build block-causal attention mask
    auto attn_mask = create_block_causal_mask(
        k_lens, q_lens, block_size_, hidden.scalar_type(), hidden.device());

    // Run through transformer blocks
    for (auto& block : blocks_) {
      hidden =
          block->forward(hidden, k_lens, q_lens, emb, block_size_, attn_mask);
    }

    // Output: AdaLN + projection
    hidden = txt_out_ada_->forward(hidden, emb, "out", "in", txt_out_norm_);
    hidden = txt_out_(hidden);

    // Extract only the Q portion (last Q_len rows)
    int64_t L_q = 0;
    for (auto ql : q_lens) {
      L_q += ql;
    }
    return hidden.narrow(0, hidden.size(0) - L_q, L_q);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_module_from_state_dicts(*loader, this);
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
