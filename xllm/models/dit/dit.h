#pragma once
#include <glog/logging.h>
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"

namespace xllm {
// DiT model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
inline torch::Tensor apply_rotary_emb(const torch::Tensor& x,
                                      const torch::Tensor& freqs_cis) {
  // assume freqs_cis is [2, S, D]，[0] is cos，[1] is sin
  auto cos = freqs_cis[0].unsqueeze(0).unsqueeze(1);  // [1, 1, 6542, 128]
  auto sin = freqs_cis[1].unsqueeze(0).unsqueeze(1);  // [1, 1, 6542, 128]
  std::vector<int64_t> reshape_shape;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    reshape_shape.push_back(x.size(i));
  }
  reshape_shape.push_back(-1);
  reshape_shape.push_back(2);
  torch::Tensor reshaped = x.reshape(reshape_shape);
  torch::Tensor x_real = reshaped.select(-1, 0);
  torch::Tensor x_imag = reshaped.select(-1, 1);
  // x_rotated = [-x_imag, x_real]
  torch::Tensor neg_x_imag = -x_imag;
  auto x_rotated = torch::stack({neg_x_imag, x_real}, -1).flatten(3);
  return (x.to(torch::kFloat32) * cos.to(torch::kFloat32) +
          x_rotated.to(torch::kFloat32) * sin.to(torch::kFloat32))
      .to(x.dtype());
}

class DiTRMSNormImpl : public torch::nn::Module {
 public:
  // Constructor: dim (normalization dimension), eps (stabilization term)
  // elementwise_affine (enable affine transform), bias (enable bias term)
  DiTRMSNormImpl(int64_t dim,
                 float eps,
                 bool elementwise_affine,
                 bool bias,
                 const torch::TensorOptions& options)
      : eps_(eps),
        elementwise_affine_(elementwise_affine),
        is_bias_(bias),
        options_(options) {
    if (elementwise_affine_) {
      weight_ = register_parameter("weight", torch::ones({dim}, options));
      if (is_bias_) {
        bias_ = register_parameter("bias", torch::zeros({dim}, options));
      }
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto input_dtype = hidden_states.dtype();

    // Compute variance in float32 for numerical stability
    auto variance = hidden_states.to(input_dtype).pow(2).mean(-1, true);
    // RMS normalization: x / sqrt(variance + eps)
    auto output = hidden_states * torch::rsqrt(variance + eps_);
    // Apply affine transform if enabled
    if (elementwise_affine_) {
      if (weight_.dtype() != torch::kFloat32) {
        output = output.to(weight_.dtype());
      }
      output = output * weight_;
      if (is_bias_) {
        output = output + bias_;
      }
    } else {
      output = output.to(input_dtype);
    }

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (elementwise_affine_) {
      auto weight = state_dict.get_tensor("weight");
      if (weight.defined()) {
        DCHECK_EQ(weight_.sizes(), weight.sizes())
            << "weight size mismatch: expected " << weight_.sizes()
            << " but got " << weight.sizes();
        weight_.data().copy_(weight.to(options_));
      }
      if (is_bias_) {
        auto bias = state_dict.get_tensor("bias");
        if (bias.defined()) {
          DCHECK_EQ(bias_.sizes(), bias.sizes())
              << "bias size mismatch: expected " << bias_.sizes() << " but got "
              << bias.sizes();
          bias_.data().copy_(bias.to(options_));
        }
      }
    }
  }

 private:
  float eps_;                // Small epsilon to avoid division by zero
  bool elementwise_affine_;  // Whether to apply learnable affine parameters
  torch::Tensor weight_;     // Learnable scale parameter
  torch::Tensor bias_;       // Learnable bias parameter (optional)
  bool is_bias_;
  torch::TensorOptions options_;
};
TORCH_MODULE(DiTRMSNorm);

class FluxSingleAttentionImpl : public torch::nn::Module {
 public:
  FluxSingleAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.dit_num_attention_heads();
    auto head_dim = model_args.dit_attention_head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;
    to_q_ = register_module("to_q",
                            DiTLinear(query_dim, out_dim, true /*has_bias*/));
    to_k_ = register_module("to_k",
                            DiTLinear(query_dim, out_dim, true /*has_bias*/));
    to_v_ = register_module("to_v",
                            DiTLinear(query_dim, out_dim, true /*has_bias*/));

    to_q_->weight.set_data(to_q_->weight.to(options_));
    to_q_->bias.set_data(to_q_->bias.to(options_));
    to_k_->weight.set_data(to_k_->weight.to(options_));
    to_k_->bias.set_data(to_k_->bias.to(options_));
    to_v_->weight.set_data(to_v_->weight.to(options_));
    to_v_->bias.set_data(to_v_->bias.to(options_));

    norm_q_ = register_module("norm_q",
                              DiTRMSNorm(head_dim,
                                         1e-6f,
                                         true /*elementwise_affine*/,
                                         false /*bias*/,
                                         options_));
    norm_k_ = register_module("norm_k",
                              DiTRMSNorm(head_dim,
                                         1e-6f,
                                         true /*elementwise_affine*/,
                                         false /*bias*/,
                                         options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& image_rotary_emb) {
    int64_t batch_size, channel, height, width;

    // Reshape 4D input to [B, seq_len, C]
    torch::Tensor hidden_states_ =
        hidden_states;  // Use copy to avoid modifying input
    batch_size = hidden_states_.size(0);

    // Self-attention: use hidden_states as context
    torch::Tensor context = hidden_states_;

    // Compute QKV projections
    torch::Tensor query = to_q_->forward(hidden_states_);
    torch::Tensor key = to_k_->forward(context);
    torch::Tensor value = to_v_->forward(context);

    // Reshape for multi-head attention
    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = heads_;
    int64_t head_dim = inner_dim / attn_heads;
    query = query.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);

    // Apply Q/K normalization if enabled
    if (norm_q_) query = norm_q_->forward(query);
    if (norm_k_) key = norm_k_->forward(key);
    // Apply rotary positional embedding
    query = apply_rotary_emb(query, image_rotary_emb);
    key = apply_rotary_emb(key, image_rotary_emb);
    // Compute scaled dot-product attention (no mask, no dropout)
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query, key, value, torch::nullopt, 0.0, false);
    attn_output = attn_output.to(query.dtype());
    return attn_output.transpose(1, 2).flatten(2);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // to_q
    const auto to_q_state_weight = state_dict.get_tensor("to_q.weight");
    if (to_q_state_weight.defined()) {
      DCHECK_EQ(to_q_->weight.sizes(), to_q_state_weight.sizes())
          << "to_q weight size mismatch: expected " << to_q_->weight.sizes()
          << " but got " << to_q_state_weight.sizes();
      to_q_->weight.data().copy_(to_q_state_weight.to(options_));
    }

    const auto to_q_state_bias = state_dict.get_tensor("to_q.bias");
    if (to_q_state_bias.defined()) {
      DCHECK_EQ(to_q_->bias.sizes(), to_q_state_bias.sizes())
          << "to_q bias size mismatch: expected " << to_q_->bias.sizes()
          << " but got " << to_q_state_bias.sizes();
      to_q_->bias.data().copy_(to_q_state_bias.to(options_));
    }
    // to_k
    const auto to_k_state_weight = state_dict.get_tensor("to_k.weight");
    if (to_k_state_weight.defined()) {
      DCHECK_EQ(to_k_->weight.sizes(), to_k_state_weight.sizes())
          << "to_k weight size mismatch: expected " << to_k_->weight.sizes()
          << " but got " << to_k_state_weight.sizes();
      to_k_->weight.data().copy_(to_k_state_weight.to(options_));
    }
    const auto to_k_state_bias = state_dict.get_tensor("to_k.bias");
    if (to_k_state_bias.defined()) {
      DCHECK_EQ(to_k_->bias.sizes(), to_k_state_bias.sizes())
          << "to_k bias size mismatch: expected " << to_k_->bias.sizes()
          << " but got " << to_k_state_bias.sizes();
      to_k_->bias.data().copy_(to_k_state_bias.to(options_));
    }

    // to_v
    const auto to_v_state_weight = state_dict.get_tensor("to_v.weight");
    if (to_v_state_weight.defined()) {
      DCHECK_EQ(to_v_->weight.sizes(), to_v_state_weight.sizes())
          << "to_v weight size mismatch: expected " << to_v_->weight.sizes()
          << " but got " << to_v_state_weight.sizes();
      to_v_->weight.data().copy_(to_v_state_weight.to(options_));
    }
    const auto to_v_state_bias = state_dict.get_tensor("to_v.bias");
    if (to_v_state_bias.defined()) {
      DCHECK_EQ(to_v_->bias.sizes(), to_v_state_bias.sizes())
          << "to_v bias size mismatch: expected " << to_v_->bias.sizes()
          << " but got " << to_v_state_bias.sizes();
      to_v_->bias.data().copy_(to_v_state_bias.to(options_));
    }
  }

 private:
  DiTLinear to_q_{nullptr};
  DiTLinear to_k_{nullptr};
  DiTLinear to_v_{nullptr};
  int64_t heads_;
  DiTRMSNorm norm_q_{nullptr};
  DiTRMSNorm norm_k_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxSingleAttention);

class FluxAttentionImpl : public torch::nn::Module {
 public:
  FluxAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.dit_num_attention_heads();
    auto head_dim = model_args.dit_attention_head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;
    auto added_kv_proj_dim = query_dim;

    to_q_ = register_module("to_q", DiTLinear(query_dim, out_dim, true));
    to_k_ = register_module("to_k", DiTLinear(query_dim, out_dim, true));
    to_v_ = register_module("to_v", DiTLinear(query_dim, out_dim, true));
    add_q_proj_ = register_module("add_q_proj",
                                  DiTLinear(added_kv_proj_dim, out_dim, true));

    add_k_proj_ = register_module("add_k_proj",
                                  DiTLinear(added_kv_proj_dim, out_dim, true));

    add_v_proj_ = register_module("add_v_proj",
                                  DiTLinear(added_kv_proj_dim, out_dim, true));

    to_out_ = register_module("to_out", DiTLinear(out_dim, query_dim, true));

    to_add_out_ = register_module("to_add_out",
                                  DiTLinear(out_dim, added_kv_proj_dim, true));

    to_q_->weight.set_data(to_q_->weight.to(options_));
    to_q_->bias.set_data(to_q_->bias.to(options_));
    to_k_->weight.set_data(to_k_->weight.to(options_));
    to_k_->bias.set_data(to_k_->bias.to(options_));

    to_v_->weight.set_data(to_v_->weight.to(options_));
    to_v_->bias.set_data(to_v_->bias.to(options_));
    add_q_proj_->weight.set_data(add_q_proj_->weight.to(options_));
    add_q_proj_->bias.set_data(add_q_proj_->bias.to(options_));
    add_k_proj_->weight.set_data(add_k_proj_->weight.to(options_));
    add_k_proj_->bias.set_data(add_k_proj_->bias.to(options_));
    add_v_proj_->weight.set_data(add_v_proj_->weight.to(options_));
    add_v_proj_->bias.set_data(add_v_proj_->bias.to(options_));
    to_out_->weight.set_data(to_out_->weight.to(options_));
    to_out_->bias.set_data(to_out_->bias.to(options_));
    to_add_out_->weight.set_data(to_add_out_->weight.to(options_));
    to_add_out_->bias.set_data(to_add_out_->bias.to(options_));

    norm_q_ = register_module("norm_q",
                              DiTRMSNorm(head_dim,
                                         1e-6f,
                                         true /*elementwise_affine*/,
                                         false /*bias*/,
                                         options_));
    norm_k_ = register_module("norm_k",
                              DiTRMSNorm(head_dim,
                                         1e-6f,
                                         true /*elementwise_affine*/,
                                         false /*bias*/,
                                         options_));
    norm_added_q_ = register_module("norm_added_q",
                                    DiTRMSNorm(head_dim,
                                               1e-6f,
                                               true /*elementwise_affine*/,
                                               false /*bias*/,
                                               options_));
    norm_added_k_ = register_module("norm_added_k",
                                    DiTRMSNorm(head_dim,
                                               1e-6f,
                                               true /*elementwise_affine*/,
                                               false /*bias*/,
                                               options_));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& image_rotary_emb) {
    int64_t input_ndim = hidden_states.dim();

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
    int64_t context_input_ndim = encoder_hidden_states.dim();
    torch::Tensor encoder_hidden_states_reshaped = encoder_hidden_states;
    if (context_input_ndim == 4) {
      auto shape = encoder_hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      encoder_hidden_states_reshaped =
          encoder_hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);
    torch::Tensor query = to_q_->forward(hidden_states_reshaped);
    torch::Tensor key = to_k_->forward(hidden_states_reshaped);
    torch::Tensor value = to_v_->forward(hidden_states_reshaped);
    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = heads_;

    int64_t head_dim = inner_dim / attn_heads;
    query = query.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    if (norm_q_) query = norm_q_->forward(query);
    if (norm_k_) key = norm_k_->forward(key);
    // encoder hidden states
    torch::Tensor encoder_hidden_states_query_proj =
        add_q_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_key_proj =
        add_k_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_value_proj =
        add_v_proj_->forward(encoder_hidden_states_reshaped);
    encoder_hidden_states_query_proj =
        encoder_hidden_states_query_proj
            .view({batch_size, -1, attn_heads, head_dim})
            .transpose(1, 2);
    encoder_hidden_states_key_proj =
        encoder_hidden_states_key_proj
            .view({batch_size, -1, attn_heads, head_dim})
            .transpose(1, 2);
    encoder_hidden_states_value_proj =
        encoder_hidden_states_value_proj
            .view({batch_size, -1, attn_heads, head_dim})
            .transpose(1, 2);
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          norm_added_q_->forward(encoder_hidden_states_query_proj);

    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          norm_added_k_->forward(encoder_hidden_states_key_proj);
    // TODO some are right some are wrong query1& key1.
    // encoder_hidden_states_query_proj
    auto query1 = torch::cat({encoder_hidden_states_query_proj, query}, 2);
    auto key1 = torch::cat({encoder_hidden_states_key_proj, key}, 2);
    auto value1 = torch::cat({encoder_hidden_states_value_proj, value}, 2);
    if (image_rotary_emb.defined()) {
      query1 = apply_rotary_emb(query1, image_rotary_emb);
      key1 = apply_rotary_emb(key1, image_rotary_emb);
    }
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query1, key1, value1, torch::nullopt, 0.0, false);

    attn_output = attn_output
                      .transpose(1, 2)  // [B, H, S, D]
                      .reshape({batch_size, -1, attn_heads * head_dim});
    attn_output = attn_output.to(query.dtype());

    int64_t encoder_length = encoder_hidden_states_reshaped.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);
    hidden_output = to_out_->forward(hidden_output);
    encoder_output = to_add_out_->forward(encoder_output);
    return std::make_tuple(hidden_output, encoder_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    //  to_q
    const auto to_q_state_weight = state_dict.get_tensor("to_q.weight");
    if (to_q_state_weight.defined()) {
      DCHECK_EQ(to_q_->weight.sizes(), to_q_state_weight.sizes())
          << "to_q weight size mismatch: expected " << to_q_->weight.sizes()
          << " but got " << to_q_state_weight.sizes();
      to_q_->weight.data().copy_(to_q_state_weight.to(options_));
    }
    const auto to_q_state_bias = state_dict.get_tensor("to_q.bias");
    if (to_q_state_bias.defined()) {
      DCHECK_EQ(to_q_->bias.sizes(), to_q_state_bias.sizes())
          << "to_q bias size mismatch: expected " << to_q_->bias.sizes()
          << " but got " << to_q_state_bias.sizes();
      to_q_->bias.data().copy_(to_q_state_bias.to(options_));
    }
    // to_k
    const auto to_k_state_weight = state_dict.get_tensor("to_k.weight");
    if (to_k_state_weight.defined()) {
      DCHECK_EQ(to_k_->weight.sizes(), to_k_state_weight.sizes())
          << "to_k weight size mismatch: expected " << to_k_->weight.sizes()
          << " but got " << to_k_state_weight.sizes();
      to_k_->weight.data().copy_(to_k_state_weight.to(options_));
    }
    const auto to_k_state_bias = state_dict.get_tensor("to_k.bias");
    if (to_k_state_bias.defined()) {
      DCHECK_EQ(to_k_->bias.sizes(), to_k_state_bias.sizes())
          << "to_k bias size mismatch: expected " << to_k_->bias.sizes()
          << " but got " << to_k_state_bias.sizes();
      to_k_->bias.data().copy_(to_k_state_bias.to(options_));
    }
    // to_v
    const auto to_v_state_weight = state_dict.get_tensor("to_v.weight");
    if (to_v_state_weight.defined()) {
      DCHECK_EQ(to_v_->weight.sizes(), to_v_state_weight.sizes())
          << "to_v weight size mismatch: expected " << to_v_->weight.sizes()
          << " but got " << to_v_state_weight.sizes();
      to_v_->weight.data().copy_(to_v_state_weight.to(options_));
    }
    const auto to_v_state_bias = state_dict.get_tensor("to_v.bias");
    if (to_v_state_bias.defined()) {
      DCHECK_EQ(to_v_->bias.sizes(), to_v_state_bias.sizes())
          << "to_v bias size mismatch: expected " << to_v_->bias.sizes()
          << " but got " << to_v_state_bias.sizes();
      to_v_->bias.data().copy_(to_v_state_bias.to(options_));
    }
    // to_out
    const auto to_out_state_weight = state_dict.get_tensor("to_out.0.weight");
    if (to_out_state_weight.defined()) {
      DCHECK_EQ(to_out_->weight.sizes(), to_out_state_weight.sizes())
          << "to_out weight size mismatch: expected " << to_out_->weight.sizes()
          << " but got " << to_out_state_weight.sizes();
      to_out_->weight.data().copy_(to_out_state_weight.to(options_));
    }
    const auto to_out_state_bias = state_dict.get_tensor("to_out.0.bias");
    if (to_out_state_bias.defined()) {
      DCHECK_EQ(to_out_->bias.sizes(), to_out_state_bias.sizes())
          << "to_out bias size mismatch: expected " << to_out_->bias.sizes()
          << " but got " << to_out_state_bias.sizes();
      to_out_->bias.data().copy_(to_out_state_bias.to(options_));
    }
    // to_add_out
    const auto to_add_out_state_weight =
        state_dict.get_tensor("to_add_out.weight");
    if (to_add_out_state_weight.defined()) {
      DCHECK_EQ(to_add_out_->weight.sizes(), to_add_out_state_weight.sizes())
          << "to_add_out weight size mismatch: expected "
          << to_add_out_->weight.sizes() << " but got "
          << to_add_out_state_weight.sizes();
      to_add_out_->weight.data().copy_(to_add_out_state_weight.to(options_));
    }
    const auto to_add_out_state_bias = state_dict.get_tensor("to_add_out.bias");
    if (to_add_out_state_bias.defined()) {
      DCHECK_EQ(to_add_out_->bias.sizes(), to_add_out_state_bias.sizes())
          << "to_add_out bias size mismatch: expected "
          << to_add_out_->bias.sizes() << " but got "
          << to_add_out_state_bias.sizes();
      to_add_out_->bias.data().copy_(to_add_out_state_bias.to(options_));
    }
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // norm_added_q
    norm_added_q_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_q."));
    // norm_added_k
    norm_added_k_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_k."));
    // add_q_proj
    const auto add_q_proj_state_weight =
        state_dict.get_tensor("add_q_proj.weight");
    if (add_q_proj_state_weight.defined()) {
      DCHECK_EQ(add_q_proj_->weight.sizes(), add_q_proj_state_weight.sizes())
          << "add_q_proj weight size mismatch: expected "
          << add_q_proj_->weight.sizes() << " but got "
          << add_q_proj_state_weight.sizes();
      add_q_proj_->weight.data().copy_(add_q_proj_state_weight.to(options_));
    }
    const auto add_q_proj_state_bias = state_dict.get_tensor("add_q_proj.bias");
    if (add_q_proj_state_bias.defined()) {
      DCHECK_EQ(add_q_proj_->bias.sizes(), add_q_proj_state_bias.sizes())
          << "add_q_proj bias size mismatch: expected "
          << add_q_proj_->bias.sizes() << " but got "
          << add_q_proj_state_bias.sizes();
      add_q_proj_->bias.data().copy_(add_q_proj_state_bias.to(options_));
    }
    // add_k_proj
    const auto add_k_proj_state_weight =
        state_dict.get_tensor("add_k_proj.weight");
    if (add_k_proj_state_weight.defined()) {
      DCHECK_EQ(add_k_proj_->weight.sizes(), add_k_proj_state_weight.sizes())
          << "add_k_proj weight size mismatch: expected "
          << add_k_proj_->weight.sizes() << " but got "
          << add_k_proj_state_weight.sizes();
      add_k_proj_->weight.data().copy_(add_k_proj_state_weight.to(options_));
    }
    const auto add_k_proj_state_bias = state_dict.get_tensor("add_k_proj.bias");
    if (add_k_proj_state_bias.defined()) {
      DCHECK_EQ(add_k_proj_->bias.sizes(), add_k_proj_state_bias.sizes())
          << "add_k_proj bias size mismatch: expected "
          << add_k_proj_->bias.sizes() << " but got "
          << add_k_proj_state_bias.sizes();
      add_k_proj_->bias.data().copy_(add_k_proj_state_bias.to(options_));
    }
    // add_v_proj
    const auto add_v_proj_state_weight =
        state_dict.get_tensor("add_v_proj.weight");
    if (add_v_proj_state_weight.defined()) {
      DCHECK_EQ(add_v_proj_->weight.sizes(), add_v_proj_state_weight.sizes())
          << "add_v_proj weight size mismatch: expected "
          << add_v_proj_->weight.sizes() << " but got "
          << add_v_proj_state_weight.sizes();
      add_v_proj_->weight.data().copy_(add_v_proj_state_weight.to(options_));
    }
    const auto add_v_proj_state_bias = state_dict.get_tensor("add_v_proj.bias");
    if (add_v_proj_state_bias.defined()) {
      DCHECK_EQ(add_v_proj_->bias.sizes(), add_v_proj_state_bias.sizes())
          << "add_v_proj bias size mismatch: expected "
          << add_v_proj_->bias.sizes() << " but got "
          << add_v_proj_state_bias.sizes();
      add_v_proj_->bias.data().copy_(add_v_proj_state_bias.to(options_));
    }
  }

 private:
  DiTLinear to_q_{nullptr};
  DiTLinear to_k_{nullptr};
  DiTLinear to_v_{nullptr};
  DiTLinear add_q_proj_{nullptr};
  DiTLinear add_k_proj_{nullptr};
  DiTLinear add_v_proj_{nullptr};
  DiTLinear to_out_{nullptr};
  DiTLinear to_add_out_{nullptr};

  DiTRMSNorm norm_q_{nullptr};
  DiTRMSNorm norm_k_{nullptr};
  DiTRMSNorm norm_added_q_{nullptr};
  DiTRMSNorm norm_added_k_{nullptr};
  int64_t heads_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxAttention);

class PixArtAlphaTextProjectionImpl : public torch::nn::Module {
 public:
  PixArtAlphaTextProjectionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    int64_t hidden_size = model_args.dit_attention_head_dim() *
                          model_args.dit_num_attention_heads();
    int64_t in_features = model_args.dit_pooled_projection_dim();
    int64_t out_dim =
        hidden_size;  //(out_features == -1) ? hidden_size : out_features;
    linear_1_ =
        register_module("linear_1", DiTLinear(in_features, hidden_size, true));

    linear_2_ =
        register_module("linear_2", DiTLinear(hidden_size, out_dim, true));

    linear_1_->weight.set_data(linear_1_->weight.to(options_));
    linear_1_->bias.set_data(linear_1_->bias.to(options_));
    linear_2_->weight.set_data(linear_2_->weight.to(options_));
    linear_2_->bias.set_data(linear_2_->bias.to(options_));
    act_1_ = register_module("act_1", torch::nn::SiLU());
  }

  torch::Tensor forward(const torch::Tensor& caption) {
    auto hidden_states = linear_1_->forward(caption);
    hidden_states = act_1_->forward(hidden_states);
    hidden_states = linear_2_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear_1
    const auto linear1_weight = state_dict.get_tensor("linear_1.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear_1_->weight.sizes())
          << "linear_1 weight size mismatch";
      linear_1_->weight.data().copy_(linear1_weight.to(options_));
    }
    const auto linear1_bias = state_dict.get_tensor("linear_1.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear_1_->bias.sizes())
          << "linear_1 bias size mismatch";
      linear_1_->bias.data().copy_(linear1_bias.to(options_));
    }
    // linear_2
    const auto linear2_weight = state_dict.get_tensor("linear_2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear_2_->weight.sizes())
          << "linear_2 weight size mismatch";
      linear_2_->weight.data().copy_(linear2_weight.to(options_));
    }
    const auto linear2_bias = state_dict.get_tensor("linear_2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear_2_->bias.sizes())
          << "linear_2 bias size mismatch";
      linear_2_->bias.data().copy_(linear2_bias.to(options_));
    }
  }

 private:
  DiTLinear linear_1_{nullptr};
  DiTLinear linear_2_{nullptr};
  torch::nn::SiLU act_1_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(PixArtAlphaTextProjection);

inline torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                            int64_t embedding_dim,
                                            bool flip_sin_to_cos = false,
                                            float downscale_freq_shift = 1.0f,
                                            float scale = 1.0f,
                                            int64_t max_period = 10000) {
  TORCH_CHECK(timesteps.dim() == 1, "Timesteps should be a 1d-array");
  int64_t half_dim = embedding_dim / 2;
  // -ln(max_period) * [0, 1, ..., half_dim-1] / (half_dim -
  // downscale_freq_shift
  torch::TensorOptions options = timesteps.options();
  auto exponent = -std::log(static_cast<float>(max_period)) *
                  torch::arange(/*start=*/0,
                                /*end=*/half_dim,
                                /*step=*/1,
                                options);
  exponent = exponent / (half_dim - downscale_freq_shift);

  // timesteps[:, None] * exp(exponent)[None, :]
  auto emb = torch::exp(exponent);                  // [half_dim]
  emb = timesteps.unsqueeze(1) * emb.unsqueeze(0);  // [N, half_dim]
  emb = scale * emb;

  // [sin(emb), cos(emb)]
  auto sin_emb = torch::sin(emb);
  auto cos_emb = torch::cos(emb);
  auto combined =
      torch::cat({sin_emb, cos_emb}, /*dim=*/-1);  // [N, 2*half_dim]

  if (flip_sin_to_cos) {
    combined = torch::cat(
        {combined.slice(
             /*dim=*/-1, /*start=*/half_dim, /*end=*/2 * half_dim),   // cos
         combined.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)},  // sin
        /*dim=*/-1);
  }

  if (embedding_dim % 2 == 1) {
    combined = torch::nn::functional::pad(
        combined, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
  }

  return combined;  // [N, embedding_dim]
}

class TimestepsImpl : public torch::nn::Module {
 public:
  TimestepsImpl(int64_t num_channels,
                bool flip_sin_to_cos,
                float downscale_freq_shift,
                int64_t scale = 1)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  num_channels_,
                                  flip_sin_to_cos_,
                                  downscale_freq_shift_,
                                  scale_,
                                  10000  // max_period
    );
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
  int64_t scale_;
};
TORCH_MODULE(Timesteps);

class TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  TimestepEmbeddingImpl(int64_t in_channels,
                        int64_t time_embed_dim,
                        int64_t out_dim,
                        int64_t cond_proj_dim,
                        bool sample_proj_bias,
                        torch::TensorOptions& options)
      : options_(options) {
    linear_1_ = register_module(
        "linear_1", DiTLinear(in_channels, time_embed_dim, sample_proj_bias));

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim == -1) ? time_embed_dim : out_dim;
    linear_2_ = register_module(
        "linear_2",
        DiTLinear(time_embed_dim, time_embed_dim_out, sample_proj_bias));
    linear_1_->weight.set_data(linear_1_->weight.to(options_));
    linear_1_->bias.set_data(linear_1_->bias.to(options_));
    linear_2_->weight.set_data(linear_2_->weight.to(options_));
    linear_2_->bias.set_data(linear_2_->bias.to(options_));
  }

  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x1 = linear_1_->forward(sample);
    x1 = act_fn_->forward(x1);
    x1 = linear_2_->forward(x1);
    return x1;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    auto linear1_weight = state_dict.get_tensor("linear_1.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear_1_->weight.sizes())
          << "linear_1 weight size mismatch";
      linear_1_->weight.data().copy_(linear1_weight.to(options_));
    }
    const auto linear1_bias = state_dict.get_tensor("linear_1.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear_1_->bias.sizes())
          << "linear_1 bias size mismatch";
      linear_1_->bias.data().copy_(linear1_bias.to(options_));
    }
    // linear2
    const auto linear2_weight = state_dict.get_tensor("linear_2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear_2_->weight.sizes())
          << "linear_2 weight size mismatch";
      linear_2_->weight.data().copy_(linear2_weight.to(options_));
    }

    const auto linear2_bias = state_dict.get_tensor("linear_2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear_2_->bias.sizes())
          << "linear_2 bias size mismatch";
      linear_2_->bias.data().copy_(linear2_bias.to(options_));
    }
  }

 private:
  DiTLinear linear_1_{nullptr};
  DiTLinear linear_2_{nullptr};
  DiTLinear cond_proj_{nullptr};
  torch::nn::SiLU post_act_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  // bool has_cond_proj_;
  torch::TensorOptions options_;
};
TORCH_MODULE(TimestepEmbedding);

class LabelEmbeddingImpl : public torch::nn::Module {
 public:
  LabelEmbeddingImpl(int64_t num_classes,
                     int64_t hidden_size,
                     float dropout_prob)
      : num_classes_(num_classes), dropout_prob_(dropout_prob) {
    bool use_cfg_embedding = dropout_prob > 0;
    embedding_table_ = register_module(
        "embedding_table",
        torch::nn::Embedding(num_classes + use_cfg_embedding, hidden_size));
  }

  torch::Tensor forward(
      torch::Tensor labels,
      c10::optional<torch::Tensor> force_drop_ids = c10::nullopt) {
    bool use_dropout = dropout_prob_ > 0;
    if ((is_training() && use_dropout) || force_drop_ids.has_value()) {
      labels = token_drop(labels, force_drop_ids);
    }

    torch::Tensor embeddings = embedding_table_->forward(labels);
    return embeddings;
  }

 private:
  torch::Tensor token_drop(
      torch::Tensor labels,
      c10::optional<torch::Tensor> force_drop_ids = c10::nullopt) {
    torch::Tensor drop_ids;
    if (!force_drop_ids.has_value()) {
      drop_ids = torch::rand({labels.size(0)}, labels.device()) < dropout_prob_;
    } else {
      drop_ids = force_drop_ids.value() == 1;
    }

    torch::Tensor mask = torch::full_like(labels, num_classes_);
    labels = torch::where(drop_ids, mask, labels);
    return labels;
  }

 private:
  torch::nn::Embedding embedding_table_{nullptr};
  int64_t num_classes_;
  float dropout_prob_;
};
TORCH_MODULE(LabelEmbedding);

class CombinedTimestepTextProjEmbeddingsImpl : public torch::nn::Module {
 public:
  CombinedTimestepTextProjEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto embedding_dim = model_args.dit_attention_head_dim() *
                         model_args.dit_num_attention_heads();
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();
    time_proj_ = Timesteps(256, true, 0.0f, 1);

    timestep_embedder_ = TimestepEmbedding(
        256,
        embedding_dim,
        -1,
        -1,
        true,
        options_);  // in_channels=256, time_embed_dim=embedding_dim
    text_embedder_ = PixArtAlphaTextProjection(context);
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_(timestep);
    auto timesteps_emb = timestep_embedder_(timesteps_proj);

    auto pooled_projections = text_embedder_(pooled_projection);
    return timesteps_emb + pooled_projections;
  }

  void load_state_dict(const StateDict& state_dict) {
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(CombinedTimestepTextProjEmbeddings);

class CombinedTimestepGuidanceTextProjEmbeddingsImpl
    : public torch::nn::Module {
 public:
  CombinedTimestepGuidanceTextProjEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()),
        time_proj_(256,
                   true,
                   0.0f,
                   1)  // num_channels=256, flip_sin_to_cos=true,
                       // downscale_freq_shift=0, scale=1
  {
    auto model_args = context.get_model_args();
    auto embedding_dim = model_args.dit_attention_head_dim() *
                         model_args.dit_num_attention_heads();
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();

    text_embedder_ = PixArtAlphaTextProjection(context);
    timestep_embedder_ = TimestepEmbedding(
        256,
        embedding_dim,
        -1,
        -1,
        true,
        options_);  // in_channels=256, time_embed_dim=embedding_dim
    guidance_embedder_ = TimestepEmbedding(
        256, embedding_dim, -1, -1, true, options_);  // in_channels=256
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_->forward(timestep);  // [N, 256]
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj);     // [N, embedding_dim]
    auto guidance_proj = time_proj_->forward(guidance);  // [N, 256]
    auto guidance_emb =
        guidance_embedder_->forward(guidance_proj);  // [N, embedding_dim]
    auto time_guidance_emb =
        timesteps_emb + guidance_emb;  // [N, embedding_dim]
    auto pooled_projections =
        text_embedder_->forward(pooled_projection);  // [N, embedding_dim]
    return time_guidance_emb + pooled_projections;   // [N, embedding_dim]
  }

  void load_state_dict(const StateDict& state_dict) {
    // guidance_embedder
    guidance_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("guidance_embedder."));
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

 private:
  TimestepEmbedding guidance_embedder_{nullptr};
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(CombinedTimestepGuidanceTextProjEmbeddings);

class CombinedTimestepLabelEmbeddingsImpl : public torch::nn::Module {
 public:
  CombinedTimestepLabelEmbeddingsImpl(int64_t num_classes,
                                      int64_t embedding_dim,
                                      float class_dropout_prob,
                                      torch::TensorOptions& options) {
    time_proj_ = register_module("time_proj", Timesteps(256, true, 1, 1));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        TimestepEmbedding(256, embedding_dim, -1, -1, true, options));
    class_embedder_ = register_module(
        "class_embedder",
        LabelEmbedding(num_classes, embedding_dim, class_dropout_prob));
  }

  torch::Tensor forward(torch::Tensor timestep, torch::Tensor class_labels) {
    torch::Tensor timesteps_proj = time_proj_(timestep);

    torch::Tensor timesteps_emb;

    timesteps_emb = timestep_embedder_(timesteps_proj);

    torch::Tensor class_emb = class_embedder_(class_labels);

    torch::Tensor conditioning = timesteps_emb + class_emb;

    return conditioning;
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  LabelEmbedding class_embedder_{nullptr};
};
TORCH_MODULE(CombinedTimestepLabelEmbeddings);

class AdaLayerNormZeroImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroImpl(int64_t embedding_dim,
                       int64_t num_embeddings,
                       bool bias,
                       torch::TensorOptions options)
      : options_(options) {
    if (num_embeddings > 0) {
      emb_ = register_module("emb",
                             CombinedTimestepLabelEmbeddings(
                                 num_embeddings, embedding_dim, 0.1, options_));
    }
    silu_ = register_module("silu", torch::nn::SiLU());

    linear_ = register_module(
        "linear", DiTLinear(embedding_dim, 6 * embedding_dim, bias));
    linear_->weight.set_data(linear_->weight.to(options_));
    linear_->bias.set_data(linear_->bias.to(options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  forward(const torch::Tensor& x,
          const torch::Tensor& timestep = torch::Tensor(),
          const torch::Tensor& class_labels = torch::Tensor(),
          const torch::Tensor& emb = torch::Tensor()) {
    torch::Tensor ada_emb = emb;
    if (!emb_.is_empty()) {
      ada_emb = emb_->forward(timestep, class_labels);
    }
    ada_emb = linear_->forward(silu_->forward(ada_emb));
    auto splits = torch::chunk(ada_emb, 6, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];
    auto shift_mlp = splits[3];
    auto scale_mlp = splits[4];
    auto gate_mlp = splits[5];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);
    return {normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight.to(options_));
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias.to(options_));
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  CombinedTimestepLabelEmbeddings emb_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZero);

class AdaLayerNormZeroSingleImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroSingleImpl(int64_t embedding_dim,
                             bool bias,
                             torch::TensorOptions& options)
      : options_(options) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear", DiTLinear(embedding_dim, 3 * embedding_dim, bias));
    linear_->weight.set_data(linear_->weight.to(options_));
    linear_->bias.set_data(linear_->bias.to(options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& emb = torch::Tensor()) {
    auto ada_emb = linear_->forward(silu_->forward(emb));
    auto splits = torch::chunk(ada_emb, 3, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);

    return {normalized_x, gate_msa};
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight.to(options_));
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias.to(options_));
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZeroSingle);

class AdaLayerNormContinuousImpl : public torch::nn::Module {
 public:
  AdaLayerNormContinuousImpl(int64_t embedding_dim,
                             int64_t conditioning_embedding_dim,
                             bool elementwise_affine,
                             double eps,
                             bool bias,
                             torch::TensorOptions& options)
      : options_(options) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        DiTLinear(conditioning_embedding_dim, 2 * embedding_dim, bias));
    linear_->weight.set_data(linear_->weight.to(options_));
    linear_->bias.set_data(linear_->bias.to(options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(elementwise_affine)
                                 .eps(eps)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& conditioning_embedding) {
    auto cond_emb = silu_->forward(conditioning_embedding);
    cond_emb = cond_emb.to(x.dtype());

    auto emb = linear_->forward(cond_emb);
    auto chunks = torch::chunk(emb, 2, 1);
    torch::Tensor scale, shift;

    scale = chunks[0];
    shift = chunks[1];
    auto x_norm = norm_->forward(x);
    return x_norm * (1 + scale).unsqueeze(1) + shift.unsqueeze(1);
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight.to(options_));
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias.to(options_));
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::string norm_type_;
  double eps_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormContinuous);

class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    auto inner_dim = dim * 4;
    auto dim_out = dim;

    // linear1
    linear1_ = register_module("linear1", DiTLinear(dim, inner_dim, true));

    // activation
    activation_ = register_module(
        "activation",
        torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
            [](const at::Tensor& x) { return torch::gelu(x, "tanh"); })));

    // linear2
    linear2_ = register_module("linear2", DiTLinear(inner_dim, dim_out, true));

    linear1_->weight.set_data(linear1_->weight.to(options_));
    linear1_->bias.set_data(linear1_->bias.to(options_));
    linear2_->weight.set_data(linear2_->weight.to(options_));
    linear2_->bias.set_data(linear2_->bias.to(options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor out = linear1_->forward(hidden_states);
    out = activation_(out);
    out = linear2_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto linear1_weight = state_dict.get_tensor("net.0.proj.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear1_->weight.sizes())
          << "linear1 weight size mismatch";
      linear1_->weight.data().copy_(linear1_weight.to(options_));
    }
    const auto linear1_bias = state_dict.get_tensor("net.0.proj.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear1_->bias.sizes())
          << "linear1 bias size mismatch";
      linear1_->bias.data().copy_(linear1_bias.to(options_));
    }
    // linear2
    const auto linear2_weight = state_dict.get_tensor("net.2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear2_->weight.sizes())
          << "linear2 weight size mismatch";
      linear2_->weight.data().copy_(linear2_weight.to(options_));
    }
    const auto linear2_bias = state_dict.get_tensor("net.2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear2_->bias.sizes())
          << "linear2 bias size mismatch";
      linear2_->bias.data().copy_(linear2_bias.to(options_));
    }
  }

 private:
  DiTLinear linear1_{nullptr};
  torch::nn::Functional activation_{nullptr};
  DiTLinear linear2_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FeedForward);

class FluxSingleTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxSingleTransformerBlockImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    mlp_hidden_dim_ = dim * 4;

    norm_ =
        register_module("norm", AdaLayerNormZeroSingle(dim, true, options_));

    int64_t mlp_out_dim = mlp_hidden_dim_;
    proj_mlp_ = register_module("proj_mlp", DiTLinear(dim, mlp_out_dim, true));

    int64_t proj_in_dim = dim + mlp_hidden_dim_;
    int64_t proj_out_dim = dim;
    proj_out_ =
        register_module("proj_out", DiTLinear(proj_in_dim, proj_out_dim, true));

    proj_mlp_->weight.set_data(proj_mlp_->weight.to(options_));
    proj_mlp_->bias.set_data(proj_mlp_->bias.to(options_));
    proj_out_->weight.set_data(proj_out_->weight.to(options_));
    proj_out_->bias.set_data(proj_out_->bias.to(options_));
    act_mlp_ =
        register_module("gelu",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    attn_ = register_module("attn", FluxSingleAttention(context));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    auto residual = hidden_states;
    auto [norm_hidden_states, gate] = norm_(hidden_states, temb);
    auto mlp_hidden_states = act_mlp_(proj_mlp_(norm_hidden_states));
    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);
    auto hidden_states_cat = torch::cat({attn_output, mlp_hidden_states}, 2);
    auto out = proj_out_(hidden_states_cat);
    out = gate.unsqueeze(1) * out;
    out = residual + out;
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    // proj_mlp
    const auto proj_mlp_weight = state_dict.get_tensor("proj_mlp.weight");
    if (proj_mlp_weight.defined()) {
      DCHECK_EQ(proj_mlp_weight.sizes(), proj_mlp_->weight.sizes())
          << "proj mlp weight size mismatch";
      proj_mlp_->weight.data().copy_(proj_mlp_weight.to(options_));
    }
    const auto proj_mlp_bias = state_dict.get_tensor("proj_mlp.bias");
    if (proj_mlp_bias.defined()) {
      DCHECK_EQ(proj_mlp_bias.sizes(), proj_mlp_->bias.sizes())
          << "proj mlp bias size mismatch";
      proj_mlp_->bias.data().copy_(proj_mlp_bias.to(options_));
    }
    // proj_out
    const auto proj_out_weight = state_dict.get_tensor("proj_out.weight");
    if (proj_out_weight.defined()) {
      DCHECK_EQ(proj_out_weight.sizes(), proj_out_->weight.sizes())
          << "proj out weight size mismatch";
      proj_out_->weight.data().copy_(proj_out_weight.to(options_));
    }
    const auto proj_out_bias = state_dict.get_tensor("proj_out.bias");
    if (proj_out_bias.defined()) {
      DCHECK_EQ(proj_out_bias.sizes(), proj_out_->bias.sizes())
          << "proj out bias size mismatch";
      proj_out_->bias.data().copy_(proj_out_bias.to(options_));
    }
  }

 private:
  AdaLayerNormZeroSingle norm_{nullptr};
  DiTLinear proj_mlp_{nullptr};
  DiTLinear proj_out_{nullptr};
  torch::nn::Functional act_mlp_{nullptr};
  FluxSingleAttention attn_{nullptr};
  int64_t mlp_hidden_dim_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxSingleTransformerBlock);

class FluxTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxTransformerBlockImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();

    auto dim = num_attention_heads * attention_head_dim;
    double eps = 1e-6;

    norm1_ = register_module("norm1",
                             AdaLayerNormZero(dim, 0, true /*bias*/, options_));

    norm1_context_ = register_module(
        "norm1_context", AdaLayerNormZero(dim, 0, true /*bias*/, options_));

    attn_ = register_module("attn", FluxAttention(context));
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_ = register_module("ff", FeedForward(context));
    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_context_ = register_module("ff_context", FeedForward(context));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    auto [norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
        norm1_(hidden_states, torch::Tensor(), torch::Tensor(), temb);
    auto [norm_encoder_hidden_states,
          c_gate_msa,
          c_shift_mlp,
          c_scale_mlp,
          c_gate_mlp] =
        norm1_context_(
            encoder_hidden_states, torch::Tensor(), torch::Tensor(), temb);
    auto [attn_output, context_attn_output] =
        attn_(norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb);
    attn_output = gate_msa.unsqueeze(1) * attn_output;
    auto new_hidden_states = hidden_states + attn_output;
    // image latent
    auto norm_hs = norm2_(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1);
    auto ff_output = ff_->forward(norm_hs);
    new_hidden_states = new_hidden_states + gate_mlp.unsqueeze(1) * ff_output;
    // context
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states + context_attn_output;
    auto norm_enc_hs = norm2_context_(new_encoder_hidden_states);
    norm_enc_hs =
        norm_enc_hs * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1);
    auto ff_context_out = ff_context_->forward(norm_enc_hs);
    new_encoder_hidden_states =
        new_encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_context_out;
    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
      new_encoder_hidden_states =
          torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
    }
    return std::make_tuple(new_hidden_states, new_encoder_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm1
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    // norm1_context
    norm1_context_->load_state_dict(
        state_dict.get_dict_with_prefix("norm1_context."));
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // ff
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    // ff_context
    ff_context_->load_state_dict(
        state_dict.get_dict_with_prefix("ff_context."));
  }

 private:
  AdaLayerNormZero norm1_{nullptr};
  AdaLayerNormZero norm1_context_{nullptr};
  FluxAttention attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  FeedForward ff_{nullptr};
  torch::nn::LayerNorm norm2_context_{nullptr};
  FeedForward ff_context_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxTransformerBlock);

class FluxTransformer2DModelImpl : public torch::nn::Module {
 public:
  FluxTransformer2DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();
    auto joint_attention_dim = model_args.dit_joint_attention_dim();
    auto axes_dims_rope = model_args.dit_axes_dims_rope();
    auto num_layers = model_args.dit_num_layers();
    auto num_single_layers = model_args.dit_num_single_layers();
    auto patch_size = model_args.dit_patch_size();
    out_channels_ = model_args.dit_in_channels();
    guidance_embeds_ = model_args.dit_guidance_embeds();

    // Initialize the transformer model components here
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());
    if (guidance_embeds_) {
      time_text_guidance_embed_ =
          register_module("time_text_guidance_embed",
                          CombinedTimestepGuidanceTextProjEmbeddings(context));
    } else {
      time_text_embed_ = register_module(
          "time_text_embed", CombinedTimestepTextProjEmbeddings(context));
    }
    context_embedder_ = register_module(
        "context_embedder", DiTLinear(joint_attention_dim, inner_dim));
    x_embedder_ =
        register_module("x_embedder", DiTLinear(out_channels_, inner_dim));
    context_embedder_->weight.set_data(context_embedder_->weight.to(options_));
    context_embedder_->bias.set_data(context_embedder_->bias.to(options_));
    x_embedder_->weight.set_data(x_embedder_->weight.to(options_));
    x_embedder_->bias.set_data(x_embedder_->bias.to(options_));
    // mm-dit block
    for (int64_t i = 0; i < num_layers; ++i) {
      transformer_blocks_->push_back(FluxTransformerBlock(context));
    }
    // single mm-dit block
    for (int64_t i = 0; i < num_single_layers; ++i) {
      single_transformer_blocks_->push_back(
          FluxSingleTransformerBlock(context));
    }
    norm_out_ =
        register_module("norm_out",
                        AdaLayerNormContinuous(inner_dim,
                                               inner_dim,
                                               false, /*elementwise_affine*/
                                               1e-6,  /*eps*/
                                               true,  /*eps*/
                                               options_));
    proj_out_ = register_module(
        "proj_out",
        DiTLinear(inner_dim, patch_size * patch_size * out_channels_, true));
    proj_out_->weight.set_data(proj_out_->weight.to(options_));
    proj_out_->bias.set_data(proj_out_->bias.to(options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    torch::Tensor temb;
    if (guidance.defined()) {
      auto guidance_scaled = guidance.to(hidden_states.dtype()) * 1000.0f;
      temb = time_text_guidance_embed_->forward(
          timestep_scaled, guidance_scaled, pooled_projections);
    } else {
      temb = time_text_embed_->forward(timestep_scaled, pooled_projections);
    }
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    bool use_step_cache = false;
    bool use_block_cache = false;

    torch::Tensor original_hidden_states = hidden_states;
    torch::Tensor original_encoder_hidden_states = encoder_hidden_states;

    // Step start: prepare inputs (hidden_states, original_hidden_states)
    TensorMap step_in_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_before(step_idx, step_in_map);
    use_step_cache = DiTCache::get_instance().on_before_step(stepin_before);

    if (!use_step_cache) {
      for (int64_t i = 0; i < transformer_blocks_->size(); ++i) {
        // Block start: prepare input (block_id)
        CacheBlockIn blockin_before(i);
        use_block_cache =
            DiTCache::get_instance().on_before_block(blockin_before);

        if (!use_block_cache) {
          auto block = transformer_blocks_[i]->as<FluxTransformerBlock>();
          auto [new_hidden, new_encoder_hidden] = block->forward(
              hidden_states, encoder_hidden_states, temb, image_rotary_emb);
          hidden_states = new_hidden;
          encoder_hidden_states = new_encoder_hidden;
        }

        // Block end: update outputs (block_id, hidden_states,
        // encoder_hidden_states, original_hidden_states,
        // original_encoder_hidden_states)
        TensorMap block_in_map = {
            {"hidden_states", hidden_states},
            {"encoder_hidden_states", encoder_hidden_states},
            {"original_hidden_states", original_hidden_states},
            {"original_encoder_hidden_states", original_encoder_hidden_states}};
        CacheBlockIn blockin_after(i, block_in_map);
        CacheBlockOut blockout_after =
            DiTCache::get_instance().on_after_block(blockin_after);

        hidden_states = blockout_after.tensors.at("hidden_states");
        encoder_hidden_states =
            blockout_after.tensors.at("encoder_hidden_states");
      }

      hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);

      for (int64_t i = 0; i < single_transformer_blocks_->size(); ++i) {
        // Block start: prepare input (block_id)
        CacheBlockIn blockin_before(i);
        use_block_cache =
            DiTCache::get_instance().on_before_block(blockin_before);

        if (!use_block_cache) {
          auto block =
              single_transformer_blocks_[i]->as<FluxSingleTransformerBlock>();
          hidden_states = block->forward(hidden_states, temb, image_rotary_emb);
        }

        // Block end: update outputs (block_id, hidden_states,
        // original_hidden_states)
        TensorMap single_block_map = {
            {"hidden_states", hidden_states},
            {"original_hidden_states", original_hidden_states}};
        CacheBlockIn blockin_after(i, single_block_map);
        CacheBlockOut blockout_after =
            DiTCache::get_instance().on_after_block(blockin_after);

        hidden_states = blockout_after.tensors.at("hidden_states");
      }

      int64_t start = encoder_hidden_states.size(1);
      int64_t length = hidden_states.size(1) - start;
      auto output_hidden =
          hidden_states.narrow(1, start, std::max(length, int64_t(0)));
      hidden_states = output_hidden;
    }

    // Step end: update outputs (hidden_states, original_hidden_states)
    TensorMap step_after_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_after(step_idx, step_after_map);
    CacheStepOut stepout_after =
        DiTCache::get_instance().on_after_step(stepin_after);
    hidden_states = stepout_after.tensors.at("hidden_states");

    auto output_hidden = norm_out_(hidden_states, temb);
    return proj_out_(output_hidden);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    // Load model parameters from the loader
    for (const auto& state_dict : loader->get_state_dicts()) {
      // context_embedder
      const auto weight = state_dict->get_tensor("context_embedder.weight");
      if (weight.defined()) {
        DCHECK_EQ(weight.sizes(), context_embedder_->weight.sizes())
            << "context_embedder weight size mismatch";
        context_embedder_->weight.data().copy_(weight.to(options_));
      }
      const auto bias = state_dict->get_tensor("context_embedder.bias");
      if (bias.defined()) {
        DCHECK_EQ(bias.sizes(), context_embedder_->bias.sizes())
            << "context_embedder bias size mismatch";
        context_embedder_->bias.data().copy_(bias.to(options_));
      }
      // x_embedder
      const auto x_weight = state_dict->get_tensor("x_embedder.weight");
      if (x_weight.defined()) {
        DCHECK_EQ(x_weight.sizes(), x_embedder_->weight.sizes())
            << "x_embedder weight size mismatch";
        x_embedder_->weight.data().copy_(x_weight.to(options_));
      }
      const auto x_bias = state_dict->get_tensor("x_embedder.bias");
      if (x_bias.defined()) {
        DCHECK_EQ(x_bias.sizes(), x_embedder_->bias.sizes())
            << "x_embedder bias size mismatch";
        x_embedder_->bias.data().copy_(x_bias.to(options_));
      }
      // time_text_embed
      if (time_text_embed_) {
        time_text_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      } else {
        time_text_guidance_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      }
      // transformer_blocks
      for (int64_t i = 0; i < transformer_blocks_->size(); ++i) {
        auto block = transformer_blocks_[i]->as<FluxTransformerBlock>();
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      // single_transformer_blocks
      for (int64_t i = 0; i < single_transformer_blocks_->size(); ++i) {
        auto block =
            single_transformer_blocks_[i]->as<FluxSingleTransformerBlock>();
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      // norm_out
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      // proj_out
      const auto proj_out_weight = state_dict->get_tensor("proj_out.weight");
      if (proj_out_weight.defined()) {
        DCHECK_EQ(proj_out_weight.sizes(), proj_out_->weight.sizes())
            << "proj_out weight size mismatch";
        proj_out_->weight.data().copy_(proj_out_weight.to(options_));
      }
      const auto proj_out_bias = state_dict->get_tensor("proj_out.bias");
      if (proj_out_bias.defined()) {
        DCHECK_EQ(proj_out_bias.sizes(), proj_out_->bias.sizes())
            << "proj_out bias size mismatch";
        proj_out_->bias.data().copy_(proj_out_bias.to(options_));
      }
    }
  }

  int64_t in_channels() { return out_channels_; }
  bool guidance_embeds() { return guidance_embeds_; }

 private:
  CombinedTimestepTextProjEmbeddings time_text_embed_{nullptr};
  CombinedTimestepGuidanceTextProjEmbeddings time_text_guidance_embed_{nullptr};
  DiTLinear context_embedder_{nullptr};
  DiTLinear x_embedder_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  AdaLayerNormContinuous norm_out_{nullptr};
  DiTLinear proj_out_{nullptr};
  bool guidance_embeds_;
  int64_t out_channels_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxTransformer2DModel);

class FluxDiTModelImpl : public torch::nn::Module {
 public:
  FluxDiTModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    flux_transformer_2d_model_ = register_module(
        "flux_transformer_2d_model", FluxTransformer2DModel(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor output =
        flux_transformer_2d_model_->forward(hidden_states_input,
                                            encoder_hidden_states_input,
                                            pooled_projections,
                                            timestep,
                                            image_rotary_emb,
                                            guidance,
                                            step_idx);
    return output;
  }
  int64_t in_channels() { return flux_transformer_2d_model_->in_channels(); }
  bool guidance_embeds() {
    return flux_transformer_2d_model_->guidance_embeds();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux_transformer_2d_model_->load_model(std::move(loader));
  }

 private:
  FluxTransformer2DModel flux_transformer_2d_model_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxDiTModel);

REGISTER_MODEL_ARGS(FluxTransformer2DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(dit_patch_size, "patch_size", 1);
  LOAD_ARG_OR(dit_in_channels, "in_channels", 64);
  LOAD_ARG_OR(dit_num_layers, "num_layers", 19);
  LOAD_ARG_OR(dit_num_single_layers, "num_single_layers", 38);
  LOAD_ARG_OR(dit_attention_head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(dit_num_attention_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(dit_joint_attention_dim, "joint_attention_dim", 4096);
  LOAD_ARG_OR(dit_pooled_projection_dim, "pooled_projection_dim", 768);
  LOAD_ARG_OR(dit_guidance_embeds, "guidance_embeds", true);
  LOAD_ARG_OR(
      dit_axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));
});
}  // namespace xllm
