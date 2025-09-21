#pragma once
#include <torch/nn/functional/linear.h>
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
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
// DiT model compatible with huggingface weights
//   ref to:
//   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
namespace xllm::hf {
namespace F = torch::nn::functional;
class LinearImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  torch::Tensor bias;
  LinearImpl(int64_t in, int64_t out, bool with_bias = true) {
    weight = register_parameter("weight", torch::empty({out, in}));
    if (with_bias) {
      bias = register_parameter("bias", torch::empty(out));
    } else {
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return F::linear(x, weight, bias);
  }
};
TORCH_MODULE(Linear);

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
                 bool elementwise_affine = true,
                 bool bias = false,
                 const at::Device& device = torch::kCPU,
                 const at::ScalarType& dtype = torch::kBFloat16)
      : eps_(eps),
        elementwise_affine_(elementwise_affine),
        is_bias_(bias),
        device_(device),
        dtype_(dtype) {
    if (elementwise_affine_) {
      weight_ =
          register_parameter("weight", torch::ones({dim}, device_).to(dtype_));
      if (is_bias_) {
        bias_ =
            register_parameter("bias", torch::zeros({dim}, device_).to(dtype_));
      }
    }
  }
  // Forward pass: applies RMS normalization
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto input_dtype = hidden_states.dtype();

    // Compute variance in float32 for numerical stability
    auto variance = hidden_states.to(dtype_).pow(2).mean(-1, true);
    // RMS normalization: x / sqrt(variance + eps)
    auto output = hidden_states * torch::rsqrt(variance + eps_);
    // Apply affine transform if enabled
    if (elementwise_affine_) {
      if (weight_.dtype() != torch::kFloat32) {
        output = output.to(weight_.dtype());
      }
      output = output * weight_.to(output.device());
      if (is_bias_) {
        output = output + bias_.to(output.device());
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
        weight_.data().copy_(weight);
        weight_.data().to(dtype_).to(device_);
      }
      if (is_bias_) {
        auto bias = state_dict.get_tensor("bias");
        if (bias.defined()) {
          DCHECK_EQ(bias_.sizes(), bias.sizes())
              << "bias size mismatch: expected " << bias_.sizes() << " but got "
              << bias.sizes();
          bias_.data().copy_(bias);
          bias_.data().to(dtype_).to(device_);
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
  at::Device device_;
  at::ScalarType dtype_;  // Data type for the parameters
};

TORCH_MODULE(DiTRMSNorm);

class FluxSingleAttentionImpl : public torch::nn::Module {
 private:
  Linear to_q_{nullptr};
  Linear to_k_{nullptr};
  Linear to_v_{nullptr};
  int64_t heads_;
  DiTRMSNorm norm_q_{nullptr};
  DiTRMSNorm norm_k_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;

 public:
  void load_state_dict(const StateDict& state_dict) {
    to_q_->to(device_);
    to_k_->to(device_);
    to_v_->to(device_);
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
  }
  FluxSingleAttentionImpl(int64_t query_dim,
                          int64_t heads,
                          int64_t head_dim,
                          int64_t out_dim,
                          const at::Device& device,
                          const at::ScalarType& dtype = torch::kBFloat16)
      : heads_(heads), device_(device), dtype_(dtype) {
    to_q_ =
        register_module("to_q", Linear(query_dim, out_dim, true /*has_bias*/));
    to_k_ =
        register_module("to_k", Linear(query_dim, out_dim, true /*has_bias*/));
    to_v_ =
        register_module("to_v", Linear(query_dim, out_dim, true /*has_bias*/));

    norm_q_ = register_module(
        "norm_q", DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
    norm_k_ = register_module(
        "norm_k", DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
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
    return attn_output.transpose(1, 2).flatten(2).to(dtype_);
  }
};
TORCH_MODULE(FluxSingleAttention);

class FluxAttentionImpl : public torch::nn::Module {
 private:
  Linear to_q_{nullptr};
  Linear to_k_{nullptr};
  Linear to_v_{nullptr};
  Linear add_q_proj_{nullptr};
  Linear add_k_proj_{nullptr};
  Linear add_v_proj_{nullptr};
  Linear to_out_{nullptr};
  Linear to_add_out_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  DiTRMSNorm norm_q_{nullptr};
  DiTRMSNorm norm_k_{nullptr};
  DiTRMSNorm norm_added_q_{nullptr};
  DiTRMSNorm norm_added_k_{nullptr};
  int64_t heads_;
  at::Device device_;
  at::ScalarType dtype_;

 public:
  void load_state_dict(const StateDict& state_dict) {
    // device management
    to_q_->to(device_);
    to_k_->to(device_);
    to_v_->to(device_);
    to_out_->to(device_);
    to_add_out_->to(device_);
    add_q_proj_->to(device_);
    add_k_proj_->to(device_);
    add_v_proj_->to(device_);
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
    // to_add_out
    const auto to_add_out_state_weight =
        state_dict.get_tensor("to_add_out.weight");
    if (to_add_out_state_weight.defined()) {
      DCHECK_EQ(to_add_out_->weight.sizes(), to_add_out_state_weight.sizes())
          << "to_add_out weight size mismatch: expected "
          << to_add_out_->weight.sizes() << " but got "
          << to_add_out_state_weight.sizes();
      to_add_out_->weight.data().copy_(to_add_out_state_weight);
      to_add_out_->weight.data().to(dtype_).to(device_);
    }
    const auto to_add_out_state_bias = state_dict.get_tensor("to_add_out.bias");
    if (to_add_out_state_bias.defined()) {
      DCHECK_EQ(to_add_out_->bias.sizes(), to_add_out_state_bias.sizes())
          << "to_add_out bias size mismatch: expected "
          << to_add_out_->bias.sizes() << " but got "
          << to_add_out_state_bias.sizes();
      to_add_out_->bias.data().copy_(to_add_out_state_bias);
      to_add_out_->bias.data().to(dtype_).to(device_);
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
      add_q_proj_->weight.data().copy_(add_q_proj_state_weight);
      add_q_proj_->weight.data().to(dtype_).to(device_);
    }
    const auto add_q_proj_state_bias = state_dict.get_tensor("add_q_proj.bias");
    if (add_q_proj_state_bias.defined()) {
      DCHECK_EQ(add_q_proj_->bias.sizes(), add_q_proj_state_bias.sizes())
          << "add_q_proj bias size mismatch: expected "
          << add_q_proj_->bias.sizes() << " but got "
          << add_q_proj_state_bias.sizes();
      add_q_proj_->bias.data().copy_(add_q_proj_state_bias);
      add_q_proj_->bias.data().to(dtype_).to(device_);
    }
    // add_k_proj
    const auto add_k_proj_state_weight =
        state_dict.get_tensor("add_k_proj.weight");
    if (add_k_proj_state_weight.defined()) {
      DCHECK_EQ(add_k_proj_->weight.sizes(), add_k_proj_state_weight.sizes())
          << "add_k_proj weight size mismatch: expected "
          << add_k_proj_->weight.sizes() << " but got "
          << add_k_proj_state_weight.sizes();
      add_k_proj_->weight.data().copy_(add_k_proj_state_weight);
      add_k_proj_->weight.data().to(dtype_).to(device_);
    }
    const auto add_k_proj_state_bias = state_dict.get_tensor("add_k_proj.bias");
    if (add_k_proj_state_bias.defined()) {
      DCHECK_EQ(add_k_proj_->bias.sizes(), add_k_proj_state_bias.sizes())
          << "add_k_proj bias size mismatch: expected "
          << add_k_proj_->bias.sizes() << " but got "
          << add_k_proj_state_bias.sizes();
      add_k_proj_->bias.data().copy_(add_k_proj_state_bias);
      add_k_proj_->bias.data().to(dtype_).to(device_);
    }
    // add_v_proj
    const auto add_v_proj_state_weight =
        state_dict.get_tensor("add_v_proj.weight");
    if (add_v_proj_state_weight.defined()) {
      DCHECK_EQ(add_v_proj_->weight.sizes(), add_v_proj_state_weight.sizes())
          << "add_v_proj weight size mismatch: expected "
          << add_v_proj_->weight.sizes() << " but got "
          << add_v_proj_state_weight.sizes();
      add_v_proj_->weight.data().copy_(add_v_proj_state_weight);
      add_v_proj_->weight.data().to(dtype_).to(device_);
    }
    const auto add_v_proj_state_bias = state_dict.get_tensor("add_v_proj.bias");
    if (add_v_proj_state_bias.defined()) {
      DCHECK_EQ(add_v_proj_->bias.sizes(), add_v_proj_state_bias.sizes())
          << "add_v_proj bias size mismatch: expected "
          << add_v_proj_->bias.sizes() << " but got "
          << add_v_proj_state_bias.sizes();
      add_v_proj_->bias.data().copy_(add_v_proj_state_bias);
      add_v_proj_->bias.data().to(dtype_).to(device_);
    }
  }
  FluxAttentionImpl(int64_t query_dim,
                    int64_t heads,
                    int64_t head_dim,
                    int64_t out_dim,
                    int64_t added_kv_proj_dim,
                    at::Device device,
                    const at::ScalarType& dtype = torch::kBFloat16)
      : heads_(heads), device_(device), dtype_(dtype) {
    to_q_ = register_module("to_q", Linear(query_dim, out_dim, true));
    to_k_ = register_module("to_k", Linear(query_dim, out_dim, true));
    to_v_ = register_module("to_v", Linear(query_dim, out_dim, true));
    add_q_proj_ =
        register_module("add_q_proj", Linear(added_kv_proj_dim, out_dim, true));

    add_k_proj_ =
        register_module("add_k_proj", Linear(added_kv_proj_dim, out_dim, true));

    add_v_proj_ =
        register_module("add_v_proj", Linear(added_kv_proj_dim, out_dim, true));

    to_out_ = register_module("to_out", Linear(out_dim, query_dim, true));

    to_add_out_ =
        register_module("to_add_out", Linear(out_dim, added_kv_proj_dim, true));

    dropout_ = register_module("dropout", torch::nn::Dropout(0.1));

    norm_q_ = register_module(
        "norm_q", DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
    norm_k_ = register_module(
        "norm_k", DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
    norm_added_q_ = register_module(
        "norm_added_q",
        DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
    norm_added_k_ = register_module(
        "norm_added_k",
        DiTRMSNorm(head_dim, 1e-6f, true, false, device_, dtype_));
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
    hidden_output = dropout_->forward(hidden_output);
    encoder_output = to_add_out_->forward(encoder_output);
    return std::make_tuple(hidden_output, encoder_output);
  }
};
TORCH_MODULE(FluxAttention);
class PixArtAlphaTextProjectionImpl : public torch::nn::Module {
 public:
  PixArtAlphaTextProjectionImpl(int64_t in_features,
                                int64_t hidden_size,
                                int64_t out_features = -1,
                                const std::string& act_fn = "gelu_tanh",
                                at::Device device = at::kCPU,
                                at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    int64_t out_dim = (out_features == -1) ? hidden_size : out_features;
    linear_1_ =
        register_module("linear_1", Linear(in_features, hidden_size, true));

    linear_2_ = register_module("linear_2", Linear(hidden_size, out_dim, true));

    act_1_ = register_module("act_1", torch::nn::SiLU());
    linear_1_->to(dtype_);
    linear_2_->to(dtype_);
    act_1_->to(dtype_);
    linear_1_->to(device_);
    linear_2_->to(device_);
    act_1_->to(device_);
  }
  void load_state_dict(const StateDict& state_dict) {
    // linear_1
    const auto linear1_weight = state_dict.get_tensor("linear_1.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear_1_->weight.sizes())
          << "linear_1 weight size mismatch";
      linear_1_->weight.data().copy_(linear1_weight);
      linear_1_->weight.data().to(dtype_).to(device_);
    }
    const auto linear1_bias = state_dict.get_tensor("linear_1.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear_1_->bias.sizes())
          << "linear_1 bias size mismatch";
      linear_1_->bias.data().copy_(linear1_bias);
      linear_1_->bias.data().to(dtype_).to(device_);
    }
    // linear_2
    const auto linear2_weight = state_dict.get_tensor("linear_2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear_2_->weight.sizes())
          << "linear_2 weight size mismatch";
      linear_2_->weight.data().copy_(linear2_weight);
      linear_2_->weight.data().to(dtype_).to(device_);
    }
    const auto linear2_bias = state_dict.get_tensor("linear_2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear_2_->bias.sizes())
          << "linear_2 bias size mismatch";
      linear_2_->bias.data().copy_(linear2_bias);
      linear_2_->bias.data().to(dtype_).to(device_);
    }
  }
  torch::Tensor forward(const torch::Tensor& caption) {
    auto hidden_states = linear_1_->forward(caption);
    hidden_states = act_1_->forward(hidden_states);
    hidden_states = linear_2_->forward(hidden_states);
    return hidden_states.to(device_).to(dtype_);
  }

 private:
  Linear linear_1_{nullptr};
  Linear linear_2_{nullptr};
  torch::nn::SiLU act_1_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(PixArtAlphaTextProjection);

inline torch::Tensor get_timestep_embedding(
    const torch::Tensor& timesteps,
    int64_t embedding_dim,
    bool flip_sin_to_cos = false,
    float downscale_freq_shift = 1.0f,
    float scale = 1.0f,
    int64_t max_period = 10000,
    at::Device device = at::kCPU,
    at::ScalarType dtype = torch::kFloat32) {
  TORCH_CHECK(timesteps.dim() == 1, "Timesteps should be a 1d-array");
  int64_t half_dim = embedding_dim / 2;
  // -ln(max_period) * [0, 1, ..., half_dim-1] / (half_dim -
  // downscale_freq_shift)
  auto exponent = -std::log(static_cast<float>(max_period)) *
                  torch::arange(/*start=*/0,
                                /*end=*/half_dim,
                                /*step=*/1,
                                torch::dtype(dtype).device(device));
  exponent = exponent / (half_dim - downscale_freq_shift);

  // timesteps[:, None] * exp(exponent)[None, :]
  auto emb = torch::exp(exponent).to(device);  // [half_dim]
  emb = timesteps.unsqueeze(1).to(dtype) *
        emb.unsqueeze(0).to(device);  // [N, half_dim]
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

  return combined.to(device).to(dtype);  // [N, embedding_dim]
}
class TimestepsImpl : public torch::nn::Module {
 public:
  TimestepsImpl(int64_t num_channels,
                bool flip_sin_to_cos,
                float downscale_freq_shift,
                int64_t scale = 1,
                at::Device device = at::kCPU,
                at::ScalarType dtype = torch::kFloat32)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale),
        device_(device),
        dtype_(dtype) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  num_channels_,
                                  flip_sin_to_cos_,
                                  downscale_freq_shift_,
                                  scale_,
                                  10000,  // max_period
                                  device_,
                                  dtype_);
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
  int64_t scale_;
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(Timesteps);
class TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  TimestepEmbeddingImpl(int64_t in_channels,
                        int64_t time_embed_dim,
                        const std::string& act_fn = "silu",
                        int64_t out_dim = -1,
                        const std::string& post_act_fn = "",
                        int64_t cond_proj_dim = -1,
                        bool sample_proj_bias = true,
                        at::Device device = at::kCPU,
                        at::ScalarType dtype = torch::kFloat32)
      : has_cond_proj_(cond_proj_dim != -1), device_(device), dtype_(dtype) {
    linear_1_ = register_module(
        "linear_1", Linear(in_channels, time_embed_dim, sample_proj_bias));

    if (cond_proj_dim != -1) {
      cond_proj_ = register_module("cond_proj",
                                   Linear(cond_proj_dim, in_channels, false));
    }

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim == -1) ? time_embed_dim : out_dim;
    linear_2_ = register_module(
        "linear_2",
        Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias));

    if (!post_act_fn.empty()) {
      post_act_ = register_module("post_act", torch::nn::SiLU());
    }
    linear_1_->to(dtype_);
    linear_2_->to(dtype_);
    if (has_cond_proj_) {
      cond_proj_->to(dtype_);
    }
    act_fn_->to(dtype_);
    if (post_act_) {
      post_act_->to(dtype_);
    }
  }
  void load_state_dict(const StateDict& state_dict) {
    linear_1_->to(device_);
    linear_2_->to(device_);
    // linear1
    auto linear1_weight = state_dict.get_tensor("linear_1.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear_1_->weight.sizes())
          << "linear_1 weight size mismatch";
      linear_1_->weight.data().copy_(linear1_weight);
      linear_1_->weight.data().to(dtype_).to(device_);
    }
    const auto linear1_bias = state_dict.get_tensor("linear_1.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear_1_->bias.sizes())
          << "linear_1 bias size mismatch";
      linear_1_->bias.data().copy_(linear1_bias);
      linear_1_->bias.data().to(dtype_).to(device_);
    }
    // linear2
    const auto linear2_weight = state_dict.get_tensor("linear_2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear_2_->weight.sizes())
          << "linear_2 weight size mismatch";
      linear_2_->weight.data().copy_(linear2_weight);
      linear_2_->weight.data().to(dtype_).to(device_);
    }
    const auto linear2_bias = state_dict.get_tensor("linear_2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear_2_->bias.sizes())
          << "linear_2 bias size mismatch";
      linear_2_->bias.data().copy_(linear2_bias);
      linear_2_->bias.data().to(dtype_).to(device_);
    }
  }
  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x = sample;
    if (has_cond_proj_ && condition.defined()) {
      x = x + cond_proj_->forward(condition);
    }
    torch::Tensor x1 = linear_1_->forward(x);
    if (act_fn_) {
      x1 = act_fn_->forward(x1);
    }
    x1 = linear_2_->forward(x1);
    if (post_act_) {
      x1 = post_act_->forward(x1);
    }
    return x1.to(device_).to(dtype_);
  }

 private:
  Linear linear_1_{nullptr};
  Linear linear_2_{nullptr};
  Linear cond_proj_{nullptr};
  torch::nn::SiLU post_act_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  bool has_cond_proj_;
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(TimestepEmbedding);
class LabelEmbeddingImpl : public torch::nn::Module {
 public:
  LabelEmbeddingImpl(int64_t num_classes,
                     int64_t hidden_size,
                     float dropout_prob,
                     at::Device device = at::kCPU,
                     at::ScalarType dtype = torch::kFloat32)
      : num_classes_(num_classes),
        dropout_prob_(dropout_prob),
        device_(device),
        dtype_(dtype) {
    bool use_cfg_embedding = dropout_prob > 0;
    embedding_table_ = register_module(
        "embedding_table",
        torch::nn::Embedding(num_classes + use_cfg_embedding, hidden_size));
  }

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
  torch::nn::Embedding embedding_table_{nullptr};
  int64_t num_classes_;
  float dropout_prob_;
  at::Device device_;
  at::ScalarType dtype_;
};

TORCH_MODULE(LabelEmbedding);

class CombinedTimestepTextProjEmbeddingsImpl : public torch::nn::Module {
 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;

 public:
  CombinedTimestepTextProjEmbeddingsImpl(int64_t embedding_dim,
                                         int64_t pooled_projection_dim,
                                         at::Device device = at::kCPU,
                                         at::ScalarType dtype = torch::kFloat32)
      : time_proj_(256,
                   true,
                   0.0f,
                   1,
                   device,
                   dtype),  // num_channels=256, flip_sin_to_cos=true,
                            // downscale_freq_shift=0, scale=1
        timestep_embedder_(
            256,
            embedding_dim,
            "silu",
            -1,
            "",
            -1,
            true,
            device,
            dtype),  // in_channels=256, time_embed_dim=embedding_dim
        text_embedder_(pooled_projection_dim,
                       embedding_dim,
                       -1,
                       "silu",
                       device,
                       dtype),
        device_(device),
        dtype_(dtype_) {}
  void load_state_dict(const StateDict& state_dict) {
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }
  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_(timestep);
    auto timesteps_emb = timestep_embedder_(
        timesteps_proj.toType(pooled_projection.dtype().toScalarType()));

    auto pooled_projections = text_embedder_(pooled_projection);
    return timesteps_emb + pooled_projections;
  }
};
TORCH_MODULE(CombinedTimestepTextProjEmbeddings);

class CombinedTimestepGuidanceTextProjEmbeddingsImpl
    : public torch::nn::Module {
 private:
  TimestepEmbedding guidance_embedder_{nullptr};
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;

 public:
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
  CombinedTimestepGuidanceTextProjEmbeddingsImpl(
      int64_t embedding_dim,
      int64_t pooled_projection_dim,
      at::Device device = at::kCPU,
      at::ScalarType dtype = torch::kFloat32)
      : time_proj_(256,
                   true,
                   0.0f,
                   1,
                   device,
                   dtype),  // num_channels=256, flip_sin_to_cos=true,
                            // downscale_freq_shift=0, scale=1
        timestep_embedder_(
            256,
            embedding_dim,
            "silu",
            -1,
            "",
            -1,
            true,
            device,
            dtype),  // in_channels=256, time_embed_dim=embedding_dim
        text_embedder_(pooled_projection_dim,
                       embedding_dim,
                       -1,
                       "silu",
                       device,
                       dtype),  // act_fn="silu"
        guidance_embedder_(256,
                           embedding_dim,
                           "silu",
                           -1,
                           "",
                           -1,
                           true,
                           device,
                           dtype),  // in_channels=256,
        device_(device),
        dtype_(dtype) {}
  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_->forward(timestep);  // [N, 256]
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj);     // [N, embedding_dim]
    auto guidance_proj = time_proj_->forward(guidance);  // [N, 256]
    auto guidance_emb = guidance_embedder_->forward(
        guidance_proj.to(dtype_));  // [N, embedding_dim]
    auto time_guidance_emb =
        timesteps_emb + guidance_emb;  // [N, embedding_dim]
    auto pooled_projections =
        text_embedder_->forward(pooled_projection);  // [N, embedding_dim]
    return time_guidance_emb + pooled_projections;   // [N, embedding_dim]
  }
};
TORCH_MODULE(CombinedTimestepGuidanceTextProjEmbeddings);
class CombinedTimestepLabelEmbeddingsImpl : public torch::nn::Module {
 public:
  CombinedTimestepLabelEmbeddingsImpl(int64_t num_classes,
                                      int64_t embedding_dim,
                                      float class_dropout_prob = 0.1,
                                      at::Device device = at::kCPU,
                                      at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    time_proj_ =
        register_module("time_proj", Timesteps(256, true, 1, 1, device, dtype));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        TimestepEmbedding(
            256, embedding_dim, "silu", -1, "", -1, true, device, dtype));
    class_embedder_ = register_module(
        "class_embedder",
        LabelEmbedding(
            num_classes, embedding_dim, class_dropout_prob, device, dtype));
  }

  torch::Tensor forward(
      torch::Tensor timestep,
      torch::Tensor class_labels,
      c10::optional<torch::Dtype> hidden_dtype = c10::nullopt) {
    torch::Tensor timesteps_proj = time_proj_(timestep);

    torch::Tensor timesteps_emb;
    if (hidden_dtype.has_value()) {
      timesteps_emb =
          timestep_embedder_(timesteps_proj.to(hidden_dtype.value()));
    } else {
      timesteps_emb = timestep_embedder_(timesteps_proj);
    }

    torch::Tensor class_emb = class_embedder_(class_labels);

    torch::Tensor conditioning = timesteps_emb + class_emb;

    return conditioning;
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  LabelEmbedding class_embedder_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(CombinedTimestepLabelEmbeddings);

class AdaLayerNormZeroImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroImpl(int64_t embedding_dim,
                       int64_t num_embeddings = 0,
                       std::string norm_type = "layer_norm",
                       bool bias = true,
                       at::Device device = at::kCPU,
                       at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    if (num_embeddings > 0) {
      emb_ = register_module(
          "emb",
          CombinedTimestepLabelEmbeddings(
              num_embeddings, embedding_dim, 0.1, device, dtype));
    }
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module("linear",
                              Linear(embedding_dim, 6 * embedding_dim, bias));

    if (norm_type == "layer_norm") {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                   .elementwise_affine(false)
                                   .eps(1e-6)));
    } else {
      TORCH_CHECK(false, "Unsupported norm_type: ", norm_type);
    }
  }
  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  forward(const torch::Tensor& x,
          const torch::Tensor& timestep = torch::Tensor(),
          const torch::Tensor& class_labels = torch::Tensor(),
          torch::Dtype hidden_dtype = torch::kFloat32,
          const torch::Tensor& emb = torch::Tensor()) {
    torch::Tensor ada_emb = emb;
    if (!emb_.is_empty()) {
      ada_emb = emb_->forward(timestep, class_labels, hidden_dtype);
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
    linear_->to(device_);
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight);
      linear_->weight.data().to(dtype_).to(device_);
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias);
      linear_->bias.data().to(dtype_).to(device_);
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  Linear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  CombinedTimestepLabelEmbeddings emb_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(AdaLayerNormZero);

class AdaLayerNormZeroSingleImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroSingleImpl(int64_t embedding_dim,
                             std::string norm_type = "layer_norm",
                             bool bias = true,
                             at::Device device = at::kCPU,
                             at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module("linear",
                              Linear(embedding_dim, 3 * embedding_dim, bias));

    if (norm_type == "layer_norm") {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                   .elementwise_affine(false)
                                   .eps(1e-6)));
    } else {
      TORCH_CHECK(false, "Unsupported norm_type: ", norm_type);
    }
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
    linear_->to(device_);
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight);
      linear_->weight.data().to(dtype_).to(device_);
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias);
      linear_->bias.data().to(dtype_).to(device_);
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  Linear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(AdaLayerNormZeroSingle);

class AdaLayerNormContinuousImpl : public torch::nn::Module {
 public:
  AdaLayerNormContinuousImpl(int64_t embedding_dim,
                             int64_t conditioning_embedding_dim,
                             bool elementwise_affine = true,
                             double eps = 1e-5,
                             bool bias = true,
                             std::string norm_type = "layer_norm",
                             at::Device device = at::kCPU,
                             at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear", Linear(conditioning_embedding_dim, 2 * embedding_dim, bias));
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
    linear_->to(device_);
    //  linear
    const auto linear_weight = state_dict.get_tensor("linear.weight");
    if (linear_weight.defined()) {
      DCHECK_EQ(linear_weight.sizes(), linear_->weight.sizes())
          << "linear weight size mismatch";
      linear_->weight.data().copy_(linear_weight);
      linear_->weight.data().to(dtype_).to(device_);
    }
    const auto linear_bias = state_dict.get_tensor("linear.bias");
    if (linear_bias.defined()) {
      DCHECK_EQ(linear_bias.sizes(), linear_->bias.sizes())
          << "linear bias size mismatch";
      linear_->bias.data().copy_(linear_bias);
      linear_->bias.data().to(dtype_).to(device_);
    }
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  Linear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::string norm_type_;
  double eps_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(AdaLayerNormContinuous);

inline torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  TORCH_CHECK(dim % 2 == 0, "Dimension must be even");

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);  // [D/2]

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);  // [S, D/2]
#if defined(USE_NPU)
  freqs_outer = freqs_outer.to(torch::kFloat32);
#endif
  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);  // [S, D/2]
    auto sin_vals = torch::sin(freqs_outer);  // [S, D/2]

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);  // [2, S, D]
  }
}

class FluxPosEmbedImpl : public torch::nn::Module {
 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;

 public:
  FluxPosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    bool is_mps = (ids.device().type() == torch::kMPS);
    torch::Dtype freqs_dtype = is_mps ? torch::kFloat32 : torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            theta_,
                                            true,  // repeat_interleave_real
                                            1,
                                            1,
                                            true,  // use_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);

    return {freqs_cos, freqs_sin};
  }
};
TORCH_MODULE(FluxPosEmbed);
class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t dim,
                  int64_t dim_out = 0,
                  int64_t mult = 4,
                  float dropout = 0.0,
                  std::string activation_fn = "geglu",
                  bool final_dropout = false,
                  int64_t inner_dim = 0,
                  bool bias = true,
                  bool out_bias = true,
                  at::Device device = torch::kCPU,
                  at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    if (inner_dim == 0) {
      inner_dim = dim * mult;
    }
    if (dim_out == 0) {
      dim_out = dim;
    }

    // linear1
    linear1_ = register_module(
        "linear1",
        Linear(dim,
               activation_fn == "geglu" || activation_fn == "swiglu" ||
                       activation_fn == "geglu-approximate"
                   ? inner_dim * 2
                   : inner_dim,
               bias));

    // activation
    if (activation_fn == "gelu") {
      activation_ = register_module(
          "activation",
          torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
              [](const at::Tensor& x) { return torch::gelu(x); })));
    } else if (activation_fn == "gelu-approximate") {
      activation_ = register_module(
          "activation",
          torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
              [](const at::Tensor& x) { return torch::gelu(x, "tanh"); })));
    } else {
      TORCH_CHECK(false, "Unsupported activation function: ", activation_fn);
    }

    // Dropout
    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));

    // linear2
    linear2_ = register_module("linear2", Linear(inner_dim, dim_out, out_bias));

    // Dropout
    if (final_dropout) {
      dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor out = linear1_->forward(hidden_states);
    out = activation_(out);
    out = dropout1_->forward(out);
    out = linear2_->forward(out);
    if (dropout2_) {
      out = dropout2_->forward(out);
    }
    return out;
  }
  void load_state_dict(const StateDict& state_dict) {
    linear1_->to(device_);
    linear2_->to(device_);
    const auto linear1_weight = state_dict.get_tensor("net.0.proj.weight");
    if (linear1_weight.defined()) {
      DCHECK_EQ(linear1_weight.sizes(), linear1_->weight.sizes())
          << "linear1 weight size mismatch";
      linear1_->weight.data().copy_(linear1_weight);
      linear1_->weight.to(dtype_).to(device_);
    }
    const auto linear1_bias = state_dict.get_tensor("net.0.proj.bias");
    if (linear1_bias.defined()) {
      DCHECK_EQ(linear1_bias.sizes(), linear1_->bias.sizes())
          << "linear1 bias size mismatch";
      linear1_->bias.data().copy_(linear1_bias);
      linear1_->bias.data().to(dtype_).to(device_);
    }
    // linear2
    const auto linear2_weight = state_dict.get_tensor("net.2.weight");
    if (linear2_weight.defined()) {
      DCHECK_EQ(linear2_weight.sizes(), linear2_->weight.sizes())
          << "linear2 weight size mismatch";
      linear2_->weight.data().copy_(linear2_weight);
      linear2_->weight.data().to(dtype_).to(device_);
    }
    const auto linear2_bias = state_dict.get_tensor("net.2.bias");
    if (linear2_bias.defined()) {
      DCHECK_EQ(linear2_bias.sizes(), linear2_->bias.sizes())
          << "linear2 bias size mismatch";
      linear2_->bias.data().copy_(linear2_bias);
      linear2_->bias.data().to(dtype_).to(device_);
    }
  }

 private:
  Linear linear1_{nullptr};
  torch::nn::Functional activation_{nullptr};
  torch::nn::Dropout dropout1_{nullptr};
  Linear linear2_{nullptr};
  torch::nn::Dropout dropout2_{nullptr};  // optional
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(FeedForward);
class FluxSingleTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxSingleTransformerBlockImpl(int64_t dim,
                                 int64_t num_attention_heads,
                                 int64_t attention_head_dim,
                                 float mlp_ratio = 4.0,
                                 at::Device device = torch::kCPU,
                                 at::ScalarType dtype = torch::kFloat32)
      : mlp_hidden_dim_(static_cast<int64_t>(dim * mlp_ratio)),
        device_(device),
        dtype_(dtype) {
    norm_ = register_module(
        "norm", AdaLayerNormZeroSingle(dim, "layer_norm", true, device, dtype));

    int64_t mlp_out_dim = mlp_hidden_dim_;
    proj_mlp_ = register_module("proj_mlp", Linear(dim, mlp_out_dim, true));

    int64_t proj_in_dim = dim + mlp_hidden_dim_;
    int64_t proj_out_dim = dim;
    proj_out_ =
        register_module("proj_out", Linear(proj_in_dim, proj_out_dim, true));

    act_mlp_ =
        register_module("gelu",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    attn_ = register_module(
        "attn",
        FluxSingleAttention(
            dim, num_attention_heads, attention_head_dim, dim, device_, dtype));
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
    if (out.scalar_type() == torch::kFloat16) {
      out = torch::clamp(out, -65504.0f, 65504.0f);
    }
    return out;
  }
  void load_state_dict(const StateDict& state_dict) {
    proj_mlp_->to(device_);
    proj_out_->to(device_);
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    // proj_mlp
    const auto proj_mlp_weight = state_dict.get_tensor("proj_mlp.weight");
    if (proj_mlp_weight.defined()) {
      DCHECK_EQ(proj_mlp_weight.sizes(), proj_mlp_->weight.sizes())
          << "proj mlp weight size mismatch";
      proj_mlp_->weight.data().copy_(proj_mlp_weight);
      proj_mlp_->weight.data().to(dtype_).to(device_);
    }
    const auto proj_mlp_bias = state_dict.get_tensor("proj_mlp.bias");
    if (proj_mlp_bias.defined()) {
      DCHECK_EQ(proj_mlp_bias.sizes(), proj_mlp_->bias.sizes())
          << "proj mlp bias size mismatch";
      proj_mlp_->bias.data().copy_(proj_mlp_bias);
      proj_mlp_->bias.data().to(dtype_).to(device_);
    }
    // proj_out
    const auto proj_out_weight = state_dict.get_tensor("proj_out.weight");
    if (proj_out_weight.defined()) {
      DCHECK_EQ(proj_out_weight.sizes(), proj_out_->weight.sizes())
          << "proj out weight size mismatch";
      proj_out_->weight.data().copy_(proj_out_weight);
      proj_out_->weight.data().to(dtype_).to(device_);
    }
    const auto proj_out_bias = state_dict.get_tensor("proj_out.bias");
    if (proj_out_bias.defined()) {
      DCHECK_EQ(proj_out_bias.sizes(), proj_out_->bias.sizes())
          << "proj out bias size mismatch";
      proj_out_->bias.data().copy_(proj_out_bias);
      proj_out_->bias.data().to(dtype_).to(device_);
    }
  }

 private:
  AdaLayerNormZeroSingle norm_{nullptr};
  Linear proj_mlp_{nullptr};
  Linear proj_out_{nullptr};
  torch::nn::Functional act_mlp_{nullptr};
  FluxSingleAttention attn_{nullptr};
  int64_t mlp_hidden_dim_;
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(FluxSingleTransformerBlock);
class FluxTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxTransformerBlockImpl(int64_t dim,
                           int64_t num_attention_heads,
                           int64_t attention_head_dim,
                           std::string qk_norm = "rms_norm",
                           double eps = 1e-6,
                           at::Device device = torch::kCPU,
                           at::ScalarType dtype = torch::kFloat32)
      : device_(device), dtype_(dtype) {
    norm1_ = register_module(
        "norm1", AdaLayerNormZero(dim, 0, "layer_norm", true, device, dtype));
    norm1_context_ = register_module(
        "norm1_context",
        AdaLayerNormZero(dim, 0, "layer_norm", true, device, dtype));
    attn_ = register_module("attn",
                            FluxAttention(dim,
                                          num_attention_heads,
                                          attention_head_dim,
                                          dim,
                                          dim,
                                          device_,
                                          dtype_));
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));
    ff_ = register_module("ff",
                          FeedForward(dim,
                                      dim,
                                      4,
                                      0,
                                      "gelu-approximate",
                                      false,
                                      0,
                                      true,
                                      true,
                                      device_,
                                      dtype_));
    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));
    ff_context_ = register_module("ff_context",
                                  FeedForward(dim,
                                              dim,
                                              4,
                                              0,
                                              "gelu-approximate",
                                              false,
                                              0,
                                              true,
                                              true,
                                              device_,
                                              dtype_));
  }
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    auto [norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
        norm1_(hidden_states, torch::Tensor(), torch::Tensor(), dtype_, temb);
    auto [norm_encoder_hidden_states,
          c_gate_msa,
          c_shift_mlp,
          c_scale_mlp,
          c_gate_mlp] = norm1_context_(encoder_hidden_states,
                                       torch::Tensor(),
                                       torch::Tensor(),
                                       dtype_,
                                       temb);
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
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(FluxTransformerBlock);

class FluxTransformer2DModelImpl : public torch::nn::Module {
 public:
  int64_t in_channels() { return out_channels_; }
  bool guidance_embeds() { return guidance_embeds_; }
  FluxTransformer2DModelImpl(int64_t patch_size = 1,
                             int64_t in_channels = 64,
                             int64_t num_layers = 19,
                             int64_t num_single_layers = 38,
                             int64_t attention_head_dim = 128,
                             int64_t num_attention_heads = 24,
                             int64_t joint_attention_dim = 4096,
                             int64_t pooled_projection_dim = 768,
                             bool guidance_embeds = true,
                             std::vector<int64_t> axes_dims_rope = {16, 56, 56},
                             at::Device device = torch::kCPU,
                             at::ScalarType dtype = torch::kFloat32)
      : out_channels_(in_channels),
        device_(device),
        dtype_(dtype),
        inner_dim_(num_attention_heads * attention_head_dim),
        guidance_embeds_(guidance_embeds)

  {
    // Initialize the transformer model components here
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());

    pos_embed_ =
        register_module("pos_embed", FluxPosEmbed(10000, axes_dims_rope));
    if (guidance_embeds) {
      time_text_guidance_embed_ = register_module(
          "time_text_guidance_embed",
          CombinedTimestepGuidanceTextProjEmbeddings(
              inner_dim_, pooled_projection_dim, device_, dtype_));
    } else {
      time_text_embed_ = register_module(
          "time_text_embed",
          CombinedTimestepTextProjEmbeddings(
              inner_dim_, pooled_projection_dim, device_, dtype_));
    }
    // TODO: check it;
    context_embedder_ = register_module(
        "context_embedder", Linear(joint_attention_dim, inner_dim_));
    x_embedder_ =
        register_module("x_embedder", Linear(in_channels, inner_dim_));
    // mm-dit block
    for (int64_t i = 0; i < num_layers; ++i) {
      transformer_blocks_->push_back(FluxTransformerBlock(inner_dim_,
                                                          num_attention_heads,
                                                          attention_head_dim,
                                                          "rms_norm",
                                                          1e-6,
                                                          device_,
                                                          dtype_));
    }
    // single mm-dit block
    for (int64_t i = 0; i < num_single_layers; ++i) {
      single_transformer_blocks_->push_back(
          FluxSingleTransformerBlock(inner_dim_,
                                     num_attention_heads,
                                     attention_head_dim,
                                     4,
                                     device_,
                                     dtype_));
    }
    norm_out_ = register_module("norm_out",
                                AdaLayerNormContinuous(inner_dim_,
                                                       inner_dim_,
                                                       false,
                                                       1e-6,
                                                       true,
                                                       "layer_norm",
                                                       device_,
                                                       dtype_));
    proj_out_ = register_module(
        "proj_out",
        Linear(inner_dim_, patch_size * patch_size * out_channels_, true));
  }
  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& img_ids,
                        const torch::Tensor& txt_ids,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor ids = torch::cat(
        {txt_ids.to(device_).to(dtype_), img_ids.to(device_).to(dtype_)}, 0);
    auto [rot_emb1, rot_emb2] = pos_embed_->forward(ids);
    rot_emb1 = rot_emb1.to(dtype_);
    rot_emb2 = rot_emb2.to(dtype_);
    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(dtype_);
    torch::Tensor hidden_states =
        x_embedder_->forward(hidden_states_input.to(device_));
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    torch::Tensor temb;
    if (guidance.defined()) {
      auto guidance_scaled = guidance.to(hidden_states.dtype()) * 1000.0f;
      time_text_guidance_embed_->to(device_);
      temb = time_text_guidance_embed_->forward(timestep_scaled.to(device_),
                                                guidance_scaled,
                                                pooled_projections.to(device_));
    } else {
      time_text_embed_->to(device_);
      temb = time_text_embed_->forward(timestep_scaled.to(device_),
                                       pooled_projections.to(device_));
    }
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input.to(device_));
    for (int64_t i = 0; i < transformer_blocks_->size(); ++i) {
      auto block = transformer_blocks_[i]->as<FluxTransformerBlock>();
      auto [new_hidden, new_encoder_hidden] = block->forward(
          hidden_states, encoder_hidden_states, temb, image_rotary_emb);
      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }
    hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);
    for (int64_t i = 0; i < single_transformer_blocks_->size(); ++i) {
      auto block =
          single_transformer_blocks_[i]->as<FluxSingleTransformerBlock>();
      hidden_states = block->forward(hidden_states, temb, image_rotary_emb);
    }
    int64_t start = encoder_hidden_states.size(1);
    int64_t length = hidden_states.size(1) - start;
    auto output_hidden =
        hidden_states.narrow(1, start, std::max(length, int64_t(0)));
    output_hidden = norm_out_(output_hidden, temb);

    return proj_out_(output_hidden);
  }
  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    context_embedder_->to(device_);
    x_embedder_->to(device_);
    proj_out_->to(device_);
    // Load model parameters from the loader
    for (const auto& state_dict : loader->get_state_dicts()) {
      // context_embedder
      const auto weight = state_dict->get_tensor("context_embedder.weight");
      if (weight.defined()) {
        DCHECK_EQ(weight.sizes(), context_embedder_->weight.sizes())
            << "context_embedder weight size mismatch";
        context_embedder_->weight.data().copy_(weight.to(dtype_).to(device_));
      }
      const auto bias = state_dict->get_tensor("context_embedder.bias");
      if (bias.defined()) {
        DCHECK_EQ(bias.sizes(), context_embedder_->bias.sizes())
            << "context_embedder bias size mismatch";
        context_embedder_->bias.data().copy_(bias.to(dtype_).to(device_));
      }
      // x_embedder
      const auto x_weight = state_dict->get_tensor("x_embedder.weight");
      if (x_weight.defined()) {
        DCHECK_EQ(x_weight.sizes(), x_embedder_->weight.sizes())
            << "x_embedder weight size mismatch";
        x_embedder_->weight.data().copy_(x_weight.to(dtype_).to(device_));
      }
      const auto x_bias = state_dict->get_tensor("x_embedder.bias");
      if (x_bias.defined()) {
        DCHECK_EQ(x_bias.sizes(), x_embedder_->bias.sizes())
            << "x_embedder bias size mismatch";
        x_embedder_->bias.data().copy_(x_bias.to(dtype_).to(device_));
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
        proj_out_->weight.data().copy_(proj_out_weight.to(dtype_).to(device_));
      }
      const auto proj_out_bias = state_dict->get_tensor("proj_out.bias");
      if (proj_out_bias.defined()) {
        DCHECK_EQ(proj_out_bias.sizes(), proj_out_->bias.sizes())
            << "proj_out bias size mismatch";
        proj_out_->bias.data().copy_(proj_out_bias.to(dtype_).to(device_));
      }
    }
  }

 private:
  int64_t out_channels_;
  int64_t inner_dim_;
  FluxPosEmbed pos_embed_{nullptr};
  CombinedTimestepTextProjEmbeddings time_text_embed_{nullptr};
  CombinedTimestepGuidanceTextProjEmbeddings time_text_guidance_embed_{nullptr};
  Linear context_embedder_{nullptr};
  Linear x_embedder_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  AdaLayerNormContinuous norm_out_{nullptr};
  Linear proj_out_{nullptr};
  at::Device device_;
  bool guidance_embeds_;
  at::ScalarType dtype_;
};
TORCH_MODULE(FluxTransformer2DModel);
class FluxDiTModelImpl : public torch::nn::Module {
 public:
  FluxDiTModelImpl(const ModelContext& context,
                   torch::Device device,
                   torch::ScalarType dtype)
      : args_(context.get_model_args()), device_(device), dtype_(dtype) {
    flux_transformer_2d_model_ = register_module(
        "flux_transformer_2d_model",
        FluxTransformer2DModel(args_.dit_patch_size(),
                               args_.dit_in_channels(),
                               args_.dit_num_layers(),
                               args_.dit_num_single_layers(),
                               args_.dit_attention_head_dim(),
                               args_.dit_num_attention_heads(),
                               args_.dit_joint_attention_dim(),
                               args_.dit_pooled_projection_dim(),
                               args_.dit_guidance_embeds(),
                               args_.dit_axes_dims_rope(),
                               device_,
                               dtype_));
    flux_transformer_2d_model_->to(dtype_);
  }
  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& img_ids,
                        const torch::Tensor& txt_ids,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor output =
        flux_transformer_2d_model_->forward(hidden_states_input,
                                            encoder_hidden_states_input,
                                            pooled_projections,
                                            timestep,
                                            img_ids,
                                            txt_ids,
                                            guidance,
                                            0);
    return output;
  }
  torch::Tensor _prepare_latent_image_ids(int64_t batch_size,
                                          int64_t height,
                                          int64_t width,
                                          torch::Device device,
                                          torch::Dtype dtype) {
    torch::Tensor latent_image_ids =
        torch::zeros({height, width, 3}, torch::dtype(dtype).device(device));
    torch::Tensor row_indices =
        torch::arange(height, torch::dtype(dtype).device(device)).unsqueeze(1);
    latent_image_ids.select(2, 1) = row_indices;
    torch::Tensor col_indices =
        torch::arange(width, torch::dtype(dtype).device(device)).unsqueeze(0);
    latent_image_ids.select(2, 2) = col_indices;
    latent_image_ids = latent_image_ids.reshape({height * width, 3});

    return latent_image_ids;
  }
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    int seed = 42;
    torch::manual_seed(seed);
    auto hidden_states = torch::randn({1, 8100, 64}, device_);
    torch::manual_seed(seed);
    auto encoder_hidden_states = torch::randn({1, 512, 4096}, device_);
    torch::manual_seed(seed);
    auto pooled_projections = torch::randn({1, 768}, device_);
    auto txt_ids = torch::zeros({512, 3}, device_);
    auto img_ids = _prepare_latent_image_ids(1, 90, 90, device_, dtype_);
    torch::Tensor timestep =
        torch::tensor({1.0f}, torch::dtype(dtype_).device(device_));
    torch::Tensor guidance =
        torch::tensor({3.5f}, torch::dtype(dtype_).device(device_));
    auto output = forward(hidden_states,
                          encoder_hidden_states,
                          pooled_projections,
                          timestep,
                          img_ids,
                          txt_ids,
                          guidance);
    return output;
  }
  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux_transformer_2d_model_->load_model(std::move(loader));
  }
  int64_t in_channels() { return flux_transformer_2d_model_->in_channels(); }
  bool guidance_embeds() {
    return flux_transformer_2d_model_->guidance_embeds();
  }

 private:
  FluxTransformer2DModel flux_transformer_2d_model_{nullptr};
  ModelArgs args_;
  at::Device device_;
  at::ScalarType dtype_;
};
TORCH_MODULE(FluxDiTModel);
REGISTER_MODEL_ARGS(FluxTransformer2DModel, [&] {
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
}  // namespace xllm::hf
