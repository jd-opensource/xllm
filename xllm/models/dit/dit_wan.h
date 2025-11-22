#pragma once
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"

namespace xllm {
inline torch::Tensor apply_rotary_emb_wan(const torch::Tensor& x,
                                          const torch::Tensor& freqs_cis) {
  auto cos_full = freqs_cis[0].unsqueeze(0).unsqueeze(1);  // [1, 1, S, D]
  auto sin_full = freqs_cis[1].unsqueeze(0).unsqueeze(1);  // [1, 1, S, D]

  int64_t D = x.size(-1);

  // Extract cos[0::2] and sin[1::2]
  auto cos_reshaped = cos_full.view({1, 1, -1, D / 2, 2});
  auto sin_reshaped = sin_full.view({1, 1, -1, D / 2, 2});
  auto cos = cos_reshaped.select(-1, 0);  // [1, 1, S, D//2]
  auto sin = sin_reshaped.select(-1, 1);  // [1, 1, S, D//2]

  // Split x into even/odd
  auto x_reshaped = x.view({x.size(0), x.size(1), x.size(2), D / 2, 2});
  auto x1 = x_reshaped.select(-1, 0);  // even indices
  auto x2 = x_reshaped.select(-1, 1);  // odd indices

  // Rotate
  auto out_even = x1.to(torch::kFloat32) * cos.to(torch::kFloat32) -
                  x2.to(torch::kFloat32) * sin.to(torch::kFloat32);
  auto out_odd = x1.to(torch::kFloat32) * sin.to(torch::kFloat32) +
                 x2.to(torch::kFloat32) * cos.to(torch::kFloat32);

  // Interleave
  auto out = torch::stack({out_even, out_odd}, -1)
                 .view({x.size(0), x.size(1), x.size(2), D});

  return out.to(x.dtype());
}

namespace F = torch::nn::functional;
class FP32LayerNormImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  torch::Tensor bias;
  FP32LayerNormImpl(const std::vector<int64_t>& normalized_shape,
                    double eps = 1e-6,
                    bool with_bias = true) {
    weight = register_parameter("weight", torch::ones(normalized_shape));
    if (with_bias) {
      bias = register_parameter("bias", torch::zeros(normalized_shape));
    } else {
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto origin_dtype = x.dtype();
    auto weight_fp32 =
        weight.defined() ? weight.to(torch::kFloat32) : torch::Tensor();
    auto bias_fp32 =
        bias.defined() ? bias.to(torch::kFloat32) : torch::Tensor();
    auto out =
        F::layer_norm(x.to(torch::kFloat32),
                      normalized_shape,
                      weight_fp32.defined() ? weight_fp32 : torch::Tensor(),
                      bias_fp32.defined() ? bias_fp32 : torch::Tensor(),
                      eps);
    return out.to(origin_dtype);
  }
};
TORCH_MODULE(FP32LayerNorm);

class WanAttentionImpl : public torch::nn::Module {
 public:
  WanAttentionImpl(int64_t dim,
                   int64_t heads,
                   int64_t dim_head,
                   double eps,
                   int64_t cross_attention_dim_head,
                   at::Device device,
                   const at::ScalarType& dtype = torch::kBFloat16)
      : heads_(heads), device_(device), dtype_(dtype) {
    int64_t inner_dim = dim_head * heads_;
    is_cross_attention_ = cross_attention_dim_head > 0;
    int64_t kv_inner_dim =
        is_cross_attention_ ? cross_attention_dim_head * heads_ : inner_dim;
    // QKV projections
    to_q_ = register_module("to_q", DiTLinear(dim, inner_dim_, true));
    to_k_ = register_module("to_k", DiTLinear(dim, kv_inner_dim, true));
    to_v_ = register_module("to_v", DiTLinear(dim, kv_inner_dim, true));
    to_out_ = register_module("to_out", DiTLinear(inner_dim, dim, true));
    dropout_ = register_module("dropout", torch::nn::Dropout(0.0));
    norm_q_ = register_module(
        "norm_q",
        DiTRMSNorm(head_dim * heads_, eps, true, false, device_, dtype_));
    norm_k_ = register_module(
        "norm_k",
        DiTRMSNorm(head_dim * heads_, eps, true, false, device_, dtype_));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::optional<torch::Tensor>& encoder_hidden_states,
      const torch::optional<std::vector<torch::Tensor>>& rotary_emb) {
    const torch::Tensor& enc_states =
        encoder_hidden_states.defined() ? encoder_hidden_states : hidden_states;

    torch::Tensor hidden_states_reshaped = hidden_states;
    if (input_ndim == 4) {
      auto shape = hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      hidden_states_reshaped =
          hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t context_input_ndim = enc_states.dim();
    torch::Tensor encoder_hidden_states_reshaped = enc_states;
    if (context_input_ndim == 4) {
      auto shape = enc_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      encoder_hidden_states_reshaped =
          enc_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }

    // QKV
    torch::Tensor query = to_q_->forward(hidden_states_reshaped);
    torch::Tensor key = to_k_->forward(encoder_hidden_states_reshaped);
    torch::Tensor value = to_v_->forward(encoder_hidden_states_reshaped);

    query = norm_q_->forward(query);
    key = norm_k_->forward(key);

    // unflatten heads
    query = query.view({query.size(0), query.size(1), heads_, -1});
    key = key.view({key.size(0), key.size(1), heads_, -1});
    value = value.view({value.size(0), value.size(1), heads_, -1});

    // rotary embedding
    if (rotary_emb.has_value()) {
      query = apply_rotary_emb_wan(query, rotary_emb.value());
      key = apply_rotary_emb_wan(key, rotary_emb.value());
    }

    torch::Tensor attn_out = torch::scaled_dot_product_attention(
        query, key, value, torch::nullopt, 0.0, false);
    attn_out = attn_out.flatten(2, 3).type_as(query);

    attn_out = to_out_->forward(attn_out);
    return attn_out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // device management
    to_q_->to(device_);
    to_k_->to(device_);
    to_v_->to(device_);
    to_out_->to(device_);
    //  to_q
    const auto to_q_state_weight = state_dict.get_tensor("to_q.weight");
    if (to_q_state_weight.defined()) {
      DCHECK_EQ(to_q_->weight.sizes(), to_q_state_weight.sizes())
          << "to_q weight size mismatch: expected " << to_q_->weight.sizes()
          << " but got " << to_q_state_weight.sizes();
      to_q_->weight.data().copy_(to_q_state_weight);
      to_q_->weight.data().to(dtype_).to(device_);
    }
    const auto to_q_state_bias = state_dict.get_tensor("to_q.bias");
    if (to_q_state_bias.defined()) {
      DCHECK_EQ(to_q_->bias.sizes(), to_q_state_bias.sizes())
          << "to_q bias size mismatch: expected " << to_q_->bias.sizes()
          << " but got " << to_q_state_bias.sizes();
      to_q_->bias.data().copy_(to_q_state_bias);
      to_q_->bias.data().to(dtype_).to(device_);
    }
    // to_k
    const auto to_k_state_weight = state_dict.get_tensor("to_k.weight");
    if (to_k_state_weight.defined()) {
      DCHECK_EQ(to_k_->weight.sizes(), to_k_state_weight.sizes())
          << "to_k weight size mismatch: expected " << to_k_->weight.sizes()
          << " but got " << to_k_state_weight.sizes();
      to_k_->weight.data().copy_(to_k_state_weight);
      to_k_->weight.data().to(dtype_).to(device_);
    }
    const auto to_k_state_bias = state_dict.get_tensor("to_k.bias");
    if (to_k_state_bias.defined()) {
      DCHECK_EQ(to_k_->bias.sizes(), to_k_state_bias.sizes())
          << "to_k bias size mismatch: expected " << to_k_->bias.sizes()
          << " but got " << to_k_state_bias.sizes();
      to_k_->bias.data().copy_(to_k_state_bias);
      to_k_->bias.data().to(dtype_).to(device_);
    }
    // to_v
    const auto to_v_state_weight = state_dict.get_tensor("to_v.weight");
    if (to_v_state_weight.defined()) {
      DCHECK_EQ(to_v_->weight.sizes(), to_v_state_weight.sizes())
          << "to_v weight size mismatch: expected " << to_v_->weight.sizes()
          << " but got " << to_v_state_weight.sizes();
      to_v_->weight.data().copy_(to_v_state_weight);
      to_v_->weight.data().to(dtype_).to(device_);
    }
    const auto to_v_state_bias = state_dict.get_tensor("to_v.bias");
    if (to_v_state_bias.defined()) {
      DCHECK_EQ(to_v_->bias.sizes(), to_v_state_bias.sizes())
          << "to_v bias size mismatch: expected " << to_v_->bias.sizes()
          << " but got " << to_v_state_bias.sizes();
      to_v_->bias.data().copy_(to_v_state_bias);
      to_v_->bias.data().to(dtype_).to(device_);
    }
    // to_out
    const auto to_out_state_weight = state_dict.get_tensor("to_out.0.weight");
    if (to_out_state_weight.defined()) {
      DCHECK_EQ(to_out_->weight.sizes(), to_out_state_weight.sizes())
          << "to_out weight size mismatch: expected " << to_out_->weight.sizes()
          << " but got " << to_out_state_weight.sizes();
      to_out_->weight.data().copy_(to_out_state_weight);
      to_out_->weight.data().to(dtype_).to(device_);
    }
    const auto to_out_state_bias = state_dict.get_tensor("to_out.0.bias");
    if (to_out_state_bias.defined()) {
      DCHECK_EQ(to_out_->bias.sizes(), to_out_state_bias.sizes())
          << "to_out bias size mismatch: expected " << to_out_->bias.sizes()
          << " but got " << to_out_state_bias.sizes();
      to_out_->bias.data().copy_(to_out_state_bias);
      to_out_->bias.data().to(dtype_).to(device_);
    }
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
  }

 private:
  int64_t heads_;
  at::Device device_;
  at::ScalarType dtype_;
  bool is_cross_attention_;
  DiTLinear to_q_{nullptr}, to_k_{nullptr}, to_v_{nullptr}, to_out_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  DiTRMSNorm norm_q_{nullptr}, norm_k_{nullptr};
};
TORCH_MODULE(WanAttention);

class WanTimeTextImageEmbeddingImpl : public torch::nn::Module {
 public:
  WanTimeTextImageEmbeddingImpl(int64_t dim,
                                int64_t time_freq_dim,
                                int64_t time_proj_dim,
                                int64_t text_embed_dim) {
    timesteps_proj_ =
        register_module("timesteps_proj", Timesteps(time_freq_dim, true, 0));
    time_embedder_ =
        register_module("time_embedder", TimestepEmbedding(time_freq_dim, dim));
    act_fn_ = register_module("act_fn", torch::nn::SiLU());
    time_proj_ =
        register_module("time_proj", torch::nn::Linear(dim, time_proj_dim));
    text_embedder_ = register_module(
        "text_embedder",
        PixArtAlphaTextProjection(text_embed_dim, dim, -1, "gelu_tanh"));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states,
      c10::optional<int64_t> timestep_seq_len = c10::nullopt) {
    torch::Tensor timestep_proj_tensor = timesteps_proj->forward(timestep);
    if (timestep_seq_len.has_value()) {
      int64_t seq_len = timestep_seq_len.value();
      int64_t batch = timestep_proj_tensor.size(0) / seq_len;
      timestep_proj_tensor =
          timestep_proj_tensor.unflatten(0, {batch, seq_len});
    }
    auto time_embedder_dtype = time_embedder->parameters().front().dtype();
    if (timestep_proj_tensor.dtype() != time_embedder_dtype &&
        time_embedder_dtype != torch::kInt8) {
      timestep_proj_tensor = timestep_proj_tensor.to(time_embedder_dtype);
    }
    torch::Tensor temb = time_embedder->forward(timestep_proj_tensor)
                             .to(encoder_hidden_states.dtype());
    torch::Tensor timestep_proj_out = time_proj->forward(act_fn->forward(temb));
    torch::Tensor encoder_hidden_states_out =
        text_embedder->forward(encoder_hidden_states);
    return std::make_tuple(temb, timestep_proj_out, encoder_hidden_states_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    // timesteps_proj
    timesteps_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("timesteps_proj."));
    // time_embedder
    time_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder."));
    // act_fn (SiLU has no parameters)
    // time_proj
    time_proj_->to(device_);
    auto time_proj_weight = state_dict.get_tensor("time_proj.weight");
    if (time_proj_weight.defined()) {
      DCHECK_EQ(time_proj_weight.sizes(), time_proj_->weight.sizes())
          << "time_proj weight size mismatch";
      time_proj_->weight.data().copy_(time_proj_weight);
      time_proj_->weight.data().to(dtype_).to(device_);
    }
    auto time_proj_bias = state_dict.get_tensor("time_proj.bias");
    if (time_proj_bias.defined()) {
      DCHECK_EQ(time_proj_bias.sizes(), time_proj_->bias.sizes())
          << "time_proj bias size mismatch";
      time_proj_->bias.data().copy_(time_proj_bias);
      time_proj_->bias.data().to(dtype_).to(device_);
    }
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

 private:
  Timesteps timesteps_proj_{nullptr};
  TimestepEmbedding time_embedder_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  torch::nn::Linear time_proj_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
};
TORCH_MODULE(WanTimeTextImageEmbedding);

class WanTransformerBlockImpl : public torch::nn::Module {
 public:
  WanTransformerBlockImpl(int64_t dim,
                          int64_t ffn_dim,
                          int64_t num_heads,
                          bool cross_attn_norm = false,
                          double eps = 1e-6,
                          at::Device device = torch::kCPU,
                          at::ScalarType dtype = torch::kFloat32)
      : cross_attn_norm_(cross_attn_norm), device_(device), dtype_(dtype) {
    norm1_ = register_module("norm1", FP32LayerNorm({dim}, eps, false));
    attn1_ = register_module(
        "attn1",
        WanAttention(dim, num_heads, dim / num_heads, eps, 0, device, dtype));
    attn2_ = register_module("attn2",
                             WanAttention(dim,
                                          num_heads,
                                          dim / num_heads,
                                          eps,
                                          dim / num_heads,
                                          device,
                                          dtype));
    if (cross_attn_norm_) {
      norm2_ = register_module("norm2", FP32LayerNorm({dim}, eps, true));
    } else {
      norm2_ = nullptr;
    }
    ffn_ =
        register_module("ffn", FeedForward(dim, ffn_dim, "gelu-approximate"));
    norm3_ = register_module("norm3", FP32LayerNorm({dim}, eps, false));
    scale_shift_table_ = register_parameter(
        "scale_shift_table", torch::randn({1, 6, dim}) / std::sqrt(dim));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& encoder_hidden_states,
                        const torch::Tensor& temb,
                        const torch::Tensor& rotary_emb) {
    torch::Tensor shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa,
        c_gate_msa;
    if (temb.dim() == 4) {
      // temb: batch_size, seq_len, 6, inner_dim
      auto scale_shift =
          scale_shift_table_.unsqueeze(0) + temb.to(torch::kFloat32);
      auto chunks = scale_shift.chunk(6, 2);
      shift_msa = chunks[0].squeeze(2);
      scale_msa = chunks[1].squeeze(2);
      gate_msa = chunks[2].squeeze(2);
      c_shift_msa = chunks[3].squeeze(2);
      c_scale_msa = chunks[4].squeeze(2);
      c_gate_msa = chunks[5].squeeze(2);
    } else {
      // temb: batch_size, 6, inner_dim
      auto scale_shift = scale_shift_table_ + temb.to(torch::kFloat32);
      auto chunks = scale_shift.chunk(6, 1);
      shift_msa = chunks[0];
      scale_msa = chunks[1];
      gate_msa = chunks[2];
      c_shift_msa = chunks[3];
      c_scale_msa = chunks[4];
      c_gate_msa = chunks[5];
    }
    // 1. Self-attention
    auto norm_hidden_states =
        norm1_->forward(hidden_states.to(torch::kFloat32)) * (1 + scale_msa) +
        shift_msa;
    norm_hidden_states = norm_hidden_states.to(hidden_states.dtype());
    auto attn_output =
        attn1_->forward(norm_hidden_states, torch::nullopt, rotary_emb);
    hidden_states = hidden_states.to(torch::kFloat32) + attn_output * gate_msa;
    hidden_states = hidden_states.to(attn_output.dtype());

    // 2. Cross-attention
    if (cross_attn_norm_ && norm2_ != nullptr) {
      norm_hidden_states = norm2_->forward(hidden_states.to(torch::kFloat32));
      norm_hidden_states = norm_hidden_states.to(hidden_states.dtype());
    } else {
      norm_hidden_states = hidden_states;
    }
    attn_output = attn2_->forward(
        norm_hidden_states, encoder_hidden_states, torch::nullopt);
    hidden_states = hidden_states + attn_output;

    // 3. Feed-forward
    norm_hidden_states =
        norm3_->forward(hidden_states.to(torch::kFloat32)) * (1 + c_scale_msa) +
        c_shift_msa;
    norm_hidden_states = norm_hidden_states.to(hidden_states.dtype());
    auto ff_output = ffn_->forward(norm_hidden_states);
    hidden_states = hidden_states.to(torch::kFloat32) +
                    ff_output.to(torch::kFloat32) * c_gate_msa;
    hidden_states = hidden_states.to(ff_output.dtype());

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm1
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    // attn1
    attn1_->load_state_dict(state_dict.get_dict_with_prefix("attn1."));
    // attn2
    attn2_->load_state_dict(state_dict.get_dict_with_prefix("attn2."));
    // norm2
    if (cross_attn_norm_ && norm2_ != nullptr) {
      norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    }
    // ffn
    ffn_->load_state_dict(state_dict.get_dict_with_prefix("ffn."));
    // norm3
    norm3_->load_state_dict(state_dict.get_dict_with_prefix("norm3."));
    // scale_shift_table
    scale_shift_table_->to(device_);
    auto scale_shift_table_weight = state_dict.get_tensor("scale_shift_table");
    if (scale_shift_table_weight.defined()) {
      DCHECK_EQ(scale_shift_table_weight.sizes(), scale_shift_table_.sizes())
          << "scale_shift_table size mismatch";
      scale_shift_table_.data().copy_(scale_shift_table_weight);
      scale_shift_table_.data().to(dtype_).to(device_);
    }
  }

 private:
  FP32LayerNorm norm1_{nullptr};
  WanAttention attn1_{nullptr};
  WanAttention attn2_{nullptr};
  FP32LayerNorm norm2_{nullptr};
  FeedForward ffn_{nullptr};
  FP32LayerNorm norm3_{nullptr};
  torch::Tensor scale_shift_table_;
  bool cross_attn_norm_;
};
TORCH_MODULE(WanTransformerBlock);

class WanTransformer3DModelImpl : public torch::nn::Module {
 public:
  WanTransformer3DModelImpl(const ModelContext& context)
      : args_(context.get_model_args()),
        device_(context.get_device()),
        dtype_(context.get_dtype()) {
    int64_t dim =
        args_.wan_num_attention_heads() * args_.wan_attention_head_dim();

    // Patch embedding (3D)
    patch_embed_ = register_module(
        "patch_embed",
        torch::nn::Conv3d(torch::nn::Conv3dOptions(args_.wan_in_channels(),
                                                   dim,
                                                   args_.wan_patch_size())
                              .stride(args_.wan_patch_size())
                              .padding(0)));

    // Text projection
    text_proj_ = register_module("text_proj",
                                 DiTLinear(args_.wan_text_dim(), dim, true));

    text_proj_->weight.set_data(text_proj_->weight.to(device_).to(dtype_));

    // Timestep embedding
    time_embed_ = register_module(
        "time_embed",
        torch::nn::Sequential(DiTLinear(args_.wan_freq_dim(), dim, true),
                              torch::nn::SiLU(),
                              DiTLinear(dim, dim, true)));

    // Transformer blocks
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < args_.wan_num_layers(); ++i) {
      blocks_->push_back(WanTransformerBlock(dim,
                                             args_.wan_num_attention_heads(),
                                             args_.wan_attention_head_dim(),
                                             args_.wan_ffn_dim(),
                                             args_.wan_cross_attn_norm(),
                                             args_.wan_qk_norm(),
                                             args_.wan_eps(),
                                             device_,
                                             dtype_));
    }

    // Final norm and projection
    norm_out_ = register_module(
        "norm_out",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(true).eps(
                args_.wan_eps())));

    proj_out_ = register_module(
        "proj_out",
        torch::nn::ConvTranspose3d(
            torch::nn::ConvTranspose3dOptions(
                dim, args_.wan_out_channels(), args_.wan_patch_size())
                .stride(args_.wan_patch_size())
                .padding(0)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& encoder_hidden_states,
                        const torch::Tensor& timestep,
                        const torch::Tensor& rotary_emb) {
    // Patch embedding
    // hidden_states: [B, C, T, H, W]
    auto x = patch_embed_->forward(hidden_states.to(device_));

    // Reshape to [B, seq_len, dim]
    int64_t batch_size = x.size(0);
    int64_t dim = x.size(1);
    int64_t t = x.size(2);
    int64_t h = x.size(3);
    int64_t w = x.size(4);

    x = x.permute({0, 2, 3, 4, 1}).contiguous();  // [B, T, H, W, dim]
    x = x.view({batch_size, t * h * w, dim});     // [B, seq_len, dim]

    // Timestep embedding
    auto t_emb = get_timestep_embedding(timestep,
                                        args_.wan_freq_dim(),
                                        false,
                                        0.0f,
                                        1.0f,
                                        10000,
                                        device_,
                                        dtype_);
    t_emb = time_embed_->forward(t_emb);

    // Add timestep embedding
    x = x + t_emb.unsqueeze(1);

    // Project text embeddings
    auto context = text_proj_->forward(encoder_hidden_states.to(device_));

    // Transformer blocks
    for (int64_t i = 0; i < blocks_->size(); ++i) {
      auto block = blocks_[i]->as<WanTransformerBlock>();
      x = block->forward(x, context, rotary_emb);
    }

    // Final norm
    x = norm_out_->forward(x);

    // Reshape back to 3D
    x = x.view({batch_size, t, h, w, dim});
    x = x.permute({0, 4, 1, 2, 3}).contiguous();  // [B, dim, T, H, W]

    // Project to output channels
    auto output = proj_out_->forward(x);

    return output;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      // patch_embed
      auto patch_weight = state_dict->get_tensor("patch_embed.weight");
      if (patch_weight.defined()) {
        patch_embed_->named_parameters()["weight"].data().copy_(
            patch_weight.to(dtype_).to(device_));
      }
      auto patch_bias = state_dict->get_tensor("patch_embed.bias");
      if (patch_bias.defined()) {
        patch_embed_->named_parameters()["bias"].data().copy_(
            patch_bias.to(dtype_).to(device_));
      }

      // text_proj
      auto text_weight = state_dict->get_tensor("text_proj.weight");
      if (text_weight.defined()) {
        text_proj_->weight.data().copy_(text_weight.to(dtype_).to(device_));
      }
      auto text_bias = state_dict->get_tensor("text_proj.bias");
      if (text_bias.defined()) {
        text_proj_->bias.data().copy_(text_bias.to(dtype_).to(device_));
      }

      // time_embed (Sequential with two Linear layers)
      auto time_0_weight = state_dict->get_tensor("time_embed.0.weight");
      if (time_0_weight.defined()) {
        time_embed_->named_modules()["0"]->as<DiTLinear>()->weight.data().copy_(
            time_0_weight.to(dtype_).to(device_));
      }
      auto time_2_weight = state_dict->get_tensor("time_embed.2.weight");
      if (time_2_weight.defined()) {
        time_embed_->named_modules()["2"]->as<DiTLinear>()->weight.data().copy_(
            time_2_weight.to(dtype_).to(device_));
      }

      // blocks
      for (int64_t i = 0; i < blocks_->size(); ++i) {
        auto block = blocks_[i]->as<WanTransformerBlock>();
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "blocks." + std::to_string(i) + "."));
      }

      // norm_out
      auto norm_weight = state_dict->get_tensor("norm_out.weight");
      if (norm_weight.defined()) {
        norm_out_->named_parameters()["weight"].data().copy_(norm_weight);
      }
      auto norm_bias = state_dict->get_tensor("norm_out.bias");
      if (norm_bias.defined()) {
        norm_out_->named_parameters()["bias"].data().copy_(norm_bias);
      }

      // proj_out
      auto proj_weight = state_dict->get_tensor("proj_out.weight");
      if (proj_weight.defined()) {
        proj_out_->named_parameters()["weight"].data().copy_(
            proj_weight.to(dtype_).to(device_));
      }
      auto proj_bias = state_dict->get_tensor("proj_out.bias");
      if (proj_bias.defined()) {
        proj_out_->named_parameters()["bias"].data().copy_(
            proj_bias.to(dtype_).to(device_));
      }
    }

    LOG(INFO) << "WanTransformer3DModel loaded successfully.";
  }

  // Generate 3D RoPE embeddings
  torch::Tensor get_3d_rotary_emb(int64_t t_len, int64_t h_len, int64_t w_len) {
    int64_t head_dim = args_.wan_attention_head_dim();
    int64_t dim_per_axis = head_dim / 3;  // Split across T, H, W

    auto options = torch::TensorOptions().dtype(dtype_).device(device_);

    // Temporal frequencies
    auto t_freqs = get_1d_rotary_freqs(t_len, dim_per_axis, options);
    // Height frequencies
    auto h_freqs = get_1d_rotary_freqs(h_len, dim_per_axis, options);
    // Width frequencies
    auto w_freqs = get_1d_rotary_freqs(w_len, dim_per_axis, options);

    // Combine into 3D grid
    auto freqs = torch::cat({t_freqs, h_freqs, w_freqs}, -1);

    // Create cos and sin
    auto cos_freqs = torch::cos(freqs);
    auto sin_freqs = torch::sin(freqs);

    return torch::stack({cos_freqs, sin_freqs}, 0);
  }

 private:
  torch::Tensor get_1d_rotary_freqs(int64_t length,
                                    int64_t dim,
                                    torch::TensorOptions options) {
    auto positions = torch::arange(length, options).unsqueeze(1);
    auto inv_freq =
        1.0 /
        torch::pow(10000.0,
                   torch::arange(0, dim, 2, options).to(torch::kFloat) / dim);

    auto freqs = positions * inv_freq.unsqueeze(0);
    return freqs;
  }

  ModelArgs args_;
  torch::nn::Conv3d patch_embed_{nullptr};
  DiTLinear text_proj_{nullptr};
  torch::nn::Sequential time_embed_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::LayerNorm norm_out_{nullptr};
  torch::nn::ConvTranspose3d proj_out_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(WanTransformer3DModel);

// ==================== WAN DiT Model Wrapper ====================
class WanDiTModelImpl : public torch::nn::Module {
 public:
  WanDiTModelImpl(const ModelContext& context)
      : args_(context.get_model_args()),
        device_(context.get_device()),
        dtype_(context.get_dtype()) {
    wan_transformer_ =
        register_module("wan_transformer", WanTransformer3DModel(context));
  }

  torch::Tensor forward(const torch::Tensor& latents,
                        const torch::Tensor& encoder_hidden_states,
                        const torch::Tensor& timestep,
                        int64_t num_frames = 1,
                        int64_t height = 64,
                        int64_t width = 64) {
    // Generate 3D rotary embeddings
    auto patch_size = args_.wan_patch_size();
    int64_t t_patches = num_frames / patch_size[0];
    int64_t h_patches = height / patch_size[1];
    int64_t w_patches = width / patch_size[2];

    auto rotary_emb =
        wan_transformer_->get_3d_rotary_emb(t_patches, h_patches, w_patches);

    // Forward pass
    auto output = wan_transformer_->forward(
        latents, encoder_hidden_states, timestep, rotary_emb);

    return output;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    wan_transformer_->load_model(std::move(loader));
  }

  int64_t in_channels() { return args_.wan_in_channels(); }
  int64_t out_channels() { return args_.wan_out_channels(); }

 private:
  WanTransformer3DModel wan_transformer_{nullptr};
  ModelArgs args_;
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(WanDiTModel);
REGISTER_MODEL_ARGS(WanTransformer3DModel, [&] {
  LOAD_ARG_OR(wan_in_channels, "in_channels", 16);
  LOAD_ARG_OR(wan_out_channels, "out_channels", 16);
  LOAD_ARG_OR(wan_num_layers, "num_layers", 30);
  LOAD_ARG_OR(wan_num_attention_heads, "num_attention_heads", 40);
  LOAD_ARG_OR(wan_attention_head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(wan_ffn_dim, "ffn_dim", 13824);
  LOAD_ARG_OR(wan_text_dim, "text_dim", 4096);
  LOAD_ARG_OR(wan_freq_dim, "freq_dim", 256);
  LOAD_ARG_OR(wan_cross_attn_norm, "cross_attn_norm", true);
  LOAD_ARG_OR(wan_qk_norm, "qk_norm", "rms_norm_across_heads");
  LOAD_ARG_OR(wan_eps, "eps", 1e-6f);
  LOAD_ARG_OR(wan_rope_max_seq_len, "rope_max_seq_len", 1024);
  LOAD_ARG_OR(wan_patch_size, "patch_size", std::vector<int64_t>({1, 2, 2}));
});

}  // namespace xllm
