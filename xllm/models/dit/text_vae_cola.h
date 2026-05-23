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

#include "core/framework/model_context.h"
#include "core/layers/common/linear.h"
#include "models/model_registry.h"
#include "transformer_cola_dit.h"
#include "transformer_longcat_audiodit.h"

namespace xllm {

// ---------------------------------------------------------------------------
// SwiGLU Activation
// ---------------------------------------------------------------------------
// Python: x, gate = x.chunk(2, dim=-1); return F.silu(gate) * x
// The input has shape (..., ffn_dim) where ffn_dim = 2 * hidden_dim.
// Output has shape (..., hidden_dim).

class SwiGLUImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(const torch::Tensor& x) {
    auto chunks = x.chunk(2, /*dim=*/-1);
    return torch::silu(chunks[1]) * chunks[0];
  }
};
TORCH_MODULE(SwiGLU);

// ---------------------------------------------------------------------------
// VAE Rotary Embedding (configurable theta)
// ---------------------------------------------------------------------------
// Same as ColaDiTRotaryEmbedding but with configurable theta.
// VAE uses theta=500000, DiT uses theta=10000.

class VAERotaryEmbeddingImpl : public torch::nn::Module {
 public:
  explicit VAERotaryEmbeddingImpl(int64_t dim, int64_t theta = 500000)
      : dim_(dim), theta_(theta) {
    auto inv_freq =
        1.0 / torch::pow(static_cast<double>(theta),
                         torch::arange(0, dim, 2, torch::kFloat32) / dim);
    register_buffer("inv_freq", inv_freq);
  }

  std::pair<torch::Tensor, torch::Tensor> get_cos_sin(int64_t length,
                                                      int64_t offset,
                                                      torch::Device device) {
    if (!inv_freq_.defined()) {
      // Recompute if buffer was not loaded from checkpoint
      inv_freq_ =
          1.0 / torch::pow(static_cast<double>(theta_),
                           torch::arange(0, dim_, 2, torch::kFloat32) / dim_);
      inv_freq_ = inv_freq_.to(device);
    }
    auto positions = torch::arange(
        offset,
        offset + length,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto freqs = positions.unsqueeze(1) * inv_freq_.unsqueeze(0);
    return {torch::cos(freqs), torch::sin(freqs)};
  }

  static std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& cos,
      const torch::Tensor& sin) {
    int64_t d = cos.size(-1);
    auto q_rot = q.narrow(-1, 0, d);
    auto q_pass = q.narrow(-1, d, q.size(-1) - d);
    auto k_rot = k.narrow(-1, 0, d);
    auto k_pass = k.narrow(-1, d, k.size(-1) - d);

    auto rotate_half = [](const torch::Tensor& x) {
      auto d = x.size(-1);
      return torch::cat({-x.narrow(-1, d / 2, d / 2), x.narrow(-1, 0, d / 2)},
                        /*dim=*/-1);
    };

    auto cos_expanded = cos.unsqueeze(1);
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
  int64_t theta_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(VAERotaryEmbedding);

// ---------------------------------------------------------------------------
// TextVAE Block (transformer block for encoder/decoder)
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py TextVAEBlock.
// Post-norm variant: x = residual + attn(norm(x)); x = residual +
// ffn(norm(x))

class TextVAEBlockImpl : public torch::nn::Module {
 public:
  TextVAEBlockImpl(int64_t dim,
                   int64_t ffn_dim,
                   int64_t num_heads,
                   int64_t shared_heads_kv,
                   int64_t rope_theta,
                   bool qk_bias = false)
      : dim_(dim),
        num_heads_(num_heads),
        head_dim_(dim / num_heads),
        shared_heads_kv_(shared_heads_kv) {
    norm_attn_ = register_module(
        "norm_attn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    int64_t kv_dim = dim / shared_heads_kv;
    qkv_proj_ = register_module(
        "qkv_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, dim + kv_dim * 2).bias(qk_bias)));
    attn_out_proj_ =
        register_module("attn_out_proj", torch::nn::Linear(dim, dim));

    // QK norm is applied to the FULL dimension before reshaping to heads.
    // This matches the Python implementation where norm is on (L, dim).
    q_norm_ = register_module(
        "q_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    k_norm_ = register_module(
        "k_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kv_dim})));

    rope_ = register_module("rope", VAERotaryEmbedding(head_dim_, rope_theta));

    norm_ffn_ = register_module(
        "norm_ffn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    // SwiGLU: proj outputs ffn_dim, SwiGLU splits into gate and up
    ffn_proj_ = register_module("ffn_proj", torch::nn::Linear(dim, ffn_dim));
    ffn_act_ = register_module("ffn_act", SwiGLU());
    // SwiGLU splits ffn_dim into two halves, so ffn_out input is ffn_dim/2
    ffn_out_ = register_module("ffn_out", torch::nn::Linear(ffn_dim / 2, dim));
  }

  // Returns K_len rows. Only the last Q_len rows are updated by attention;
  // prefix rows pass through unchanged so residuals work on K_len uniformly.
  torch::Tensor forward(const torch::Tensor& x,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false) {
    int64_t L_k = x.size(0);  // full K-side length
    int64_t B = k_lens.size();
    int64_t kv_dim = dim_ / shared_heads_kv_;

    // Compute total Q_len
    int64_t L_q = 0;
    for (auto ql : q_lens) {
      L_q += ql;
    }

    // --- Post-norm attention (post_norm=True in config) ---
    // When post_norm=True: h = x (no pre-norm), norm is applied AFTER attention
    auto h = x;
    {
      auto s = x.cpu();
      LOG(INFO) << "VAE-blk: input: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // QKV projection over full K_len
    auto qkv = qkv_proj_(h);
    auto qkv_chunks = qkv.split({dim_, kv_dim, kv_dim}, /*dim=*/-1);
    auto q = qkv_chunks[0];  // (L_k, dim)
    auto k = qkv_chunks[1];  // (L_k, kv_dim)
    auto v = qkv_chunks[2];  // (L_k, kv_dim)

    // QK normalization on FULL dimension (before reshape to heads).
    // This matches the Python implementation.
    auto dtype = q.dtype();
    q = q_norm_(q).to(dtype);
    k = k_norm_(k).to(dtype);

    // Reshape K to (L_k, heads, head_dim)
    k = k.reshape({L_k, num_heads_ / shared_heads_kv_, head_dim_});

    // Apply RoPE to full K_len for K (positions [0, K_len))
    torch::Tensor cos_k_all, sin_k_all;
    {
      std::vector<torch::Tensor> cos_list, sin_list;
      for (int64_t i = 0; i < B; ++i) {
        auto [cos, sin] = rope_->get_cos_sin(k_lens[i], 0, x.device());
        cos_list.push_back(cos);
        sin_list.push_back(sin);
      }
      cos_k_all = torch::cat(cos_list, /*dim=*/0);
      sin_k_all = torch::cat(sin_list, /*dim=*/0);
    }
    k = VAERotaryEmbeddingImpl::apply_rotary_emb_single(
        k, cos_k_all, sin_k_all);

    // Apply RoPE to full K_len for Q (positions [0, K_len) — last L_q rows
    // correspond to positions [K_len-L_q, K_len) matching Python q_offset)
    q = q.reshape({L_k, num_heads_, head_dim_});
    q = VAERotaryEmbeddingImpl::apply_rotary_emb_single(
        q, cos_k_all, sin_k_all);

    // Extract Q as last L_q rows for attention
    auto q_for_attn = q.narrow(0, L_k - L_q, L_q);  // (L_q, heads, head_dim)

    // Expand K/V heads if shared_heads_kv > 1
    if (shared_heads_kv_ > 1) {
      k = k.repeat({1, shared_heads_kv_, 1});
    }

    // Attention: (1, heads, L, head_dim)
    auto q_na = q_for_attn.permute({1, 0, 2}).unsqueeze(0);
    auto k_na = k.permute({1, 0, 2}).unsqueeze(0);
    auto v_reshaped =
        v.reshape({L_k, num_heads_ / shared_heads_kv_, head_dim_});
    if (shared_heads_kv_ > 1) {
      v_reshaped = v_reshaped.repeat({1, shared_heads_kv_, 1});
    }
    auto v_na = v_reshaped.permute({1, 0, 2}).unsqueeze(0);

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
      {
        auto aw = attn_weight.cpu();
        LOG(INFO) << "VAE-blk: attn_weight: shape=[" << aw.size(0) << ","
                  << aw.size(1) << "," << aw.size(2) << "," << aw.size(3)
                  << "], mean=" << aw.mean().item<float>()
                  << ", max=" << aw.max().item<float>() << ", entropy="
                  << (-aw.log().mul(aw).sum(-1).mean()).item<float>();
      }
      attn_out = attn_weight.matmul(v_na.to(torch::kBFloat16));
    }
    attn_out = attn_out.squeeze(0).permute({1, 0, 2}).reshape({L_q, dim_});
    auto attn_result = attn_out_proj_(attn_out.to(dtype));
    {
      auto s = attn_result.cpu();
      LOG(INFO) << "VAE-blk: attn_result: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // Post-norm (post_norm=True):
    //   attn: x = norm_attn(x) + attn          (norm BEFORE attn)
    //   ffn:  x = residual + norm_ffn(ffn_out(ffn_act(ffn_proj(x))))  (norm
    //   AFTER ffn)
    // Only Q positions get residual + norm; K-only positions pass through.
    auto q_part = x.narrow(0, L_k - L_q, L_q).clone();
    // Attention sublayer: norm BEFORE attention, then add
    q_part = norm_attn_(q_part) + attn_result;
    {
      auto s = q_part.cpu();
      LOG(INFO) << "VAE-blk: after norm_attn: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // FFN sublayer: proj → act → out → norm, then add residual
    auto residual = q_part;
    auto ffn_hidden = ffn_proj_(q_part);
    ffn_hidden = ffn_act_(ffn_hidden);
    auto ffn_result = ffn_out_(ffn_hidden);
    q_part = residual + norm_ffn_(ffn_result);
    {
      auto s = q_part.cpu();
      LOG(INFO) << "VAE-blk: after norm_ffn: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // Reassemble: K-only positions unchanged, Q positions updated
    if (L_q < L_k) {
      auto result = x.clone();
      result.narrow(0, L_k - L_q, L_q).copy_(q_part);
      return result;
    }
    return q_part;
  }

 private:
  int64_t dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t shared_heads_kv_;
  torch::nn::LayerNorm norm_attn_{nullptr};
  torch::nn::Linear qkv_proj_{nullptr};
  torch::nn::Linear attn_out_proj_{nullptr};
  torch::nn::LayerNorm q_norm_{nullptr};
  torch::nn::LayerNorm k_norm_{nullptr};
  VAERotaryEmbedding rope_{nullptr};

  torch::nn::LayerNorm norm_ffn_{nullptr};
  torch::nn::Linear ffn_proj_{nullptr};
  SwiGLU ffn_act_{nullptr};
  torch::nn::Linear ffn_out_{nullptr};
};
TORCH_MODULE(TextVAEBlock);

// ---------------------------------------------------------------------------
// ColaTextVAE Encoder
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py ColaTextVAEModel.encode()
// Token embedding -> Conv1d patchification -> transformer blocks ->
// final_layer -> final_norm -> per-sample latents

class ColaTextVAEEncoderImpl : public torch::nn::Module {
 public:
  ColaTextVAEEncoderImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t vae_dim = args.vae_dim();
    dim_ = vae_dim;
    int64_t vae_num_heads = args.vae_num_heads();
    int64_t ffn_dim = args.ffn_dim();
    int64_t shared_heads_kv = args.shared_heads_kv();
    int64_t vae_rope_theta = args.vae_rope_theta();
    int64_t latent_dim = args.latent_dim();
    int64_t patch_size = args.vae_patch_size();
    int64_t encoder_num_blocks = args.encoder_num_blocks();
    bool use_variation = args.use_variation();
    bool encoder_last_ln = args.encoder_last_ln();
    bool qk_bias = args.qk_bias();
    int64_t vocab_size = args.vocab_size();

    wte_ =
        register_module("wte", torch::nn::Embedding(vocab_size + 1, vae_dim));
    patch_embedder_ = register_module(
        "patch_embedder",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(vae_dim, vae_dim, patch_size)
                              .stride(patch_size)));

    blocks_.reserve(encoder_num_blocks);
    for (int64_t i = 0; i < encoder_num_blocks; ++i) {
      auto block = TextVAEBlock(vae_dim,
                                ffn_dim,
                                vae_num_heads,
                                shared_heads_kv,
                                vae_rope_theta,
                                qk_bias);
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    if (use_variation) {
      final_layer_ = register_module(
          "final_layer", torch::nn::Linear(vae_dim, latent_dim * 2));
      if (encoder_last_ln) {
        final_norm_ = register_module(
            "final_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})
                                     .elementwise_affine(false)));
      }
    } else {
      final_layer_ = register_module("final_layer",
                                     torch::nn::Linear(vae_dim, latent_dim));
      final_norm_ = register_module(
          "final_norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})
                                   .elementwise_affine(false)));
    }

    use_variation_ = use_variation;
    encoder_last_ln_ = encoder_last_ln;
    block_size_ = args.vae_block_size();
    patch_size_ = patch_size;
  }

  // Encode per-sample input_ids into per-sample latents.
  // Returns concatenated latents (L_total, latent_dim) and per-sample lengths.
  std::pair<torch::Tensor, std::vector<int64_t>> encode(
      const std::vector<torch::Tensor>& input_ids_list) {
    torch::NoGradGuard no_grad;

    // Embedding + Conv1d patchification per sample
    std::vector<torch::Tensor> per_sample;
    per_sample.reserve(input_ids_list.size());
    for (const auto& ids : input_ids_list) {
      if (ids.numel() == 0) {
        LOG(WARNING) << "ColaTextVAEEncoder: empty input_ids, skipping";
        continue;
      }
      auto x = wte_(ids.unsqueeze(0));      // (1, L_i, dim)
      x = x.permute({0, 2, 1});             // (1, dim, L_i)
      x = patch_embedder_(x);               // (1, dim, n_i)
      x = x.permute({0, 2, 1}).squeeze(0);  // (n_i, dim)
      per_sample.push_back(x);
    }

    if (per_sample.empty()) {
      LOG(WARNING) << "ColaTextVAEEncoder: all input samples are empty";
      return {torch::zeros({0, dim_}, wte_->weight.options()),
              std::vector<int64_t>{}};
    }

    // Build txt_shape for NA layout
    std::vector<int64_t> sample_lens;
    sample_lens.reserve(per_sample.size());
    for (const auto& t : per_sample) {
      sample_lens.push_back(t.size(0));
    }
    int64_t L_total = 0;
    for (auto l : sample_lens) L_total += l;

    // Concatenate all samples
    auto x = torch::cat(per_sample, /*dim=*/0);  // (L_total, dim)
    // Cast to model dtype (bf16) to match Python's autocast behavior.
    x = x.to(wte_->weight.dtype());

    // Build block-causal attention mask
    torch::Tensor attn_mask;
    if (block_size_ > 0) {
      attn_mask = create_block_causal_mask(
          sample_lens, sample_lens, block_size_, x.scalar_type(), x.device());
    }

    // Run through transformer blocks
    for (auto& block : blocks_) {
      x = block->forward(x, sample_lens, sample_lens, block_size_, attn_mask);
    }

    // Final projection
    x = final_layer_(x);
    if (encoder_last_ln_ && use_variation_) {
      // Split mean/logvar, apply norm to mean only
      auto chunks = x.chunk(2, /*dim=*/-1);
      auto mean = final_norm_(chunks[0]);
      x = torch::cat({mean, chunks[1]}, /*dim=*/-1);
    } else if (final_norm_) {
      x = final_norm_(x);
    }

    return {x, sample_lens};
  }

 private:
  int64_t dim_ = 0;
  torch::nn::Embedding wte_{nullptr};
  torch::nn::Conv1d patch_embedder_{nullptr};
  std::vector<TextVAEBlock> blocks_;
  torch::nn::Linear final_layer_{nullptr};
  torch::nn::LayerNorm final_norm_{nullptr};
  bool use_variation_ = true;
  bool encoder_last_ln_ = true;
  int64_t block_size_ = 0;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEEncoder);

// ---------------------------------------------------------------------------
// ColaTextVAE Decoder
//-----------
// Reference: modeling_cola_vae.py ColaTextVAEModel.decode()
// Latent projection -> transformer blocks -> unpatch -> final_norm ->
// vocab projection

class ColaTextVAEDecoderImpl : public torch::nn::Module {
 public:
  ColaTextVAEDecoderImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t vae_dim = args.vae_dim();
    int64_t vae_num_heads = args.vae_num_heads();
    int64_t ffn_dim = args.ffn_dim();
    int64_t shared_heads_kv = args.shared_heads_kv();
    int64_t vae_rope_theta = args.vae_rope_theta();
    int64_t latent_dim = args.latent_dim();
    int64_t patch_size = args.vae_patch_size();
    int64_t decoder_num_blocks = args.decoder_num_blocks();
    int64_t vocab_size = args.vocab_size();
    bool qk_bias = args.qk_bias();

    in_layer_ =
        register_module("in_layer", torch::nn::Linear(latent_dim, vae_dim));

    blocks_.reserve(decoder_num_blocks);
    for (int64_t i = 0; i < decoder_num_blocks; ++i) {
      auto block = TextVAEBlock(vae_dim,
                                ffn_dim,
                                vae_num_heads,
                                shared_heads_kv,
                                vae_rope_theta,
                                qk_bias);
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    unpatch_layer_ = register_module(
        "unpatch_layer", torch::nn::Linear(vae_dim, patch_size * vae_dim));
    final_norm_ = register_module(
        "final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({vae_dim})));
    final_layer_ =
        register_module("final_layer", torch::nn::Linear(vae_dim, vocab_size));

    block_size_ = args.vae_block_size();
    patch_size_ = patch_size;
  }

  // Decode latents into vocabulary logits.
  // z: (L_total, latent_dim)
  // Returns: (1, L_total * patch_size, vocab_size)
  torch::Tensor decode(const torch::Tensor& z,
                       const std::vector<int64_t>& k_lens,
                       const std::vector<int64_t>& q_lens,
                       bool update_kv = false) {
    torch::NoGradGuard no_grad;

    // Cast to model dtype (bf16) to match Python's autocast behavior.
    auto model_dtype = parameters().front().dtype();
    auto hidden = in_layer_(z.to(model_dtype));  // (L_total, dim)
    {
      auto s = hidden.cpu();
      LOG(INFO) << "VAE-dec: after in_layer: shape=[" << s.size(0) << ","
                << s.size(1) << "], mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // Build block-causal attention mask
    torch::Tensor attn_mask;
    if (block_size_ > 0) {
      attn_mask = create_block_causal_mask(
          k_lens, q_lens, block_size_, hidden.scalar_type(), hidden.device());
    }

    // Run through transformer blocks (return K_len; only Q portion updated)
    for (int bi = 0; bi < static_cast<int>(blocks_.size()); ++bi) {
      hidden = blocks_[bi]->forward(
          hidden, k_lens, q_lens, block_size_, attn_mask, update_kv);
    }
    {
      auto s = hidden.cpu();
      LOG(INFO) << "VAE-dec: after blocks: shape=[" << s.size(0) << ","
                << s.size(1) << "], mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }

    // Extract Q portion (last L_q rows)
    int64_t L_q = 0;
    for (auto ql : q_lens) {
      L_q += ql;
    }
    hidden = hidden.narrow(0, hidden.size(0) - L_q, L_q);

    // Unpatch: (L, dim) -> (L, dim*patch_size) -> (L*patch_size, dim)
    hidden = unpatch_layer_(hidden);  // (L, dim*patch_size)
    {
      auto s = hidden.cpu();
      LOG(INFO) << "VAE-dec: after unpatch: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }
    hidden = hidden.reshape({L_q * patch_size_, -1});  // (L*ps, dim)
    hidden = final_norm_(hidden);
    {
      auto s = hidden.cpu();
      LOG(INFO) << "VAE-dec: after final_norm: mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
    }
    hidden = final_layer_(hidden);  // (L*ps, vocab)
    {
      auto s = hidden.cpu();
      LOG(INFO) << "VAE-dec: after final: shape=[" << s.size(0) << ","
                << s.size(1) << "], mean=" << s.mean().item<float>()
                << ", std=" << s.std().item<float>();
      // Log argmax tokens
      auto argmax = s.argmax(/*dim=*/-1);
      std::string toks;
      for (int64_t i = 0; i < std::min(argmax.size(0), (int64_t)16); ++i) {
        toks += std::to_string(argmax[i].item<int64_t>()) + " ";
      }
      LOG(INFO) << "VAE-dec: argmax tokens: [" << toks << "]";
    }
    return hidden.unsqueeze(0);  // (1, L*ps, vocab)
  }

 private:
  torch::nn::Linear in_layer_{nullptr};
  std::vector<TextVAEBlock> blocks_;
  torch::nn::Linear unpatch_layer_{nullptr};
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear final_layer_{nullptr};
  int64_t block_size_ = 0;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEDecoder);

// ---------------------------------------------------------------------------
// ColaTextVAEModel (main model combining encoder + decoder)
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py ColaTextVAEModel

class ColaTextVAEModelImpl : public torch::nn::Module {
 public:
  explicit ColaTextVAEModelImpl(const ModelContext& ctx) {
    encoder_ = register_module("encoder", ColaTextVAEEncoder(ctx));
    decoder_ = register_module("decoder", ColaTextVAEDecoder(ctx));

    const auto& args = ctx.get_model_args();
    scaling_factor_ = args.scaling_factor();
    shifting_factor_ = args.shifting_factor();
    use_variation_ = args.use_variation();
    patch_size_ = args.vae_patch_size();
  }

  // Encode input_ids to latents with optional scaling/shifting.
  // Returns per-sample latents and their lengths.
  std::pair<torch::Tensor, std::vector<int64_t>> encode(
      const std::vector<torch::Tensor>& input_ids_list) {
    auto [latents, sample_lens] = encoder_->encode(input_ids_list);

    // Apply DiagonalGaussianDistribution if using variation
    if (use_variation_) {
      // Split into mean and logvar, sample from distribution
      auto chunks = latents.chunk(2, /*dim=*/-1);
      auto mean = chunks[0];
      auto logvar = torch::clamp(chunks[1], -30.0f, 20.0f);
      // Use mean (deterministic) for inference
      latents = mean;
    }

    // Apply scaling and shifting: z = (z - shifting) * scaling
    if (scaling_factor_ != 1.0f || shifting_factor_ != 0.0f) {
      latents = (latents - shifting_factor_) * scaling_factor_;
    }

    return {latents, sample_lens};
  }

  // Decode latents to vocabulary logits.
  torch::Tensor decode(const torch::Tensor& z,
                       const std::vector<int64_t>& k_lens,
                       const std::vector<int64_t>& q_lens,
                       bool update_kv = false) {
    return decoder_->decode(z, k_lens, q_lens, update_kv);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_module_from_state_dicts(*loader, encoder_.ptr().get(), "encoder.");
    load_module_from_state_dicts(*loader, decoder_.ptr().get(), "decoder.");
  }

  float scaling_factor() const { return scaling_factor_; }
  float shifting_factor() const { return shifting_factor_; }
  int64_t patch_size() const { return patch_size_; }

 private:
  ColaTextVAEEncoder encoder_{nullptr};
  ColaTextVAEDecoder decoder_{nullptr};
  float scaling_factor_ = 1.0f;
  float shifting_factor_ = 0.0f;
  bool use_variation_ = true;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEModel);

// ---------------------------------------------------------------------------
// REGISTER_MODEL_ARGS for ColaTextVAE config.json
// ---------------------------------------------------------------------------

REGISTER_MODEL_ARGS(cola_text_vae, [&] {
  LOAD_ARG_OR(model_type, "model_type", "cola_text_vae");
  LOAD_ARG(vocab_size, "vocab_size");
  LOAD_ARG(vae_dim, "dim");
  LOAD_ARG(vae_num_heads, "num_heads");
  LOAD_ARG(encoder_num_blocks, "encoder_num_blocks");
  LOAD_ARG(decoder_num_blocks, "decoder_num_blocks");
  LOAD_ARG(ffn_dim, "ffn_dim");
  LOAD_ARG(latent_dim, "latent_dim");
  LOAD_ARG(shared_heads_kv, "shared_heads_kv");
  LOAD_ARG_OR(vae_rope_theta, "rope_theta", 500000);
  LOAD_ARG_OR(vae_block_size, "block_size", 4);
  LOAD_ARG_OR(vae_patch_size, "patch_size", 1);
  LOAD_ARG_OR(encoder_last_ln, "encoder_last_ln", true);
  LOAD_ARG_OR(use_variation, "use_variation", true);
  LOAD_ARG_OR(qk_bias, "qk_bias", false);
  LOAD_ARG_OR(scaling_factor, "scaling_factor", 1.0f);
  LOAD_ARG_OR(shifting_factor, "shifting_factor", 0.0f);
});

}  // namespace xllm
