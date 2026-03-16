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

#include <glog/logging.h>
#include <torch/nn/functional.h>

#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>

#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_qwen3_omni_code2wav_transformer_layer_impl.h"
#include "framework/state_dict/state_dict.h"
namespace xllm {

// using torch::indexing::None;
// using ISlice = torch::indexing::Slice;

class Qwen3OmniCausalTransConvNextImpl : public torch::nn::Module {
 public:
  Qwen3OmniCausalTransConvNextImpl(const ModelContext& context,
                                   int64_t in_channels,
                                   int64_t out_channels,
                                   int64_t kernel_size,
                                   int64_t stride = 1) {
    conv_ =
        torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(
                                       in_channels, out_channels, kernel_size)
                                       .stride(stride));
    register_module("conv", conv_);
    auto pad = (kernel_size - stride);
    left_pad_ = static_cast<int64_t>(std::ceil(static_cast<double>(pad)));
    pad = left_pad_;
    right_pad_ = left_pad_;
  }
  void load_state_dict(const StateDict& state_dict) {
    auto conv_weight = state_dict.get_tensor("weight");
    auto conv_bias = state_dict.get_tensor("bias");
    if (conv_weight.defined()) {
      conv_->weight.data().copy_(conv_weight);
      conv_weight_loaded_ = true;
    }

    if (conv_bias.defined()) {
      conv_->bias.data().copy_(conv_bias);
      conv_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(conv_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

  void verify_loaded_bias(const std::string& prefix) const {
    CHECK(conv_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv_->forward(x);
    int64_t T = x.size(-1);
    int64_t valid_length = T - left_pad_ - right_pad_;
    auto hidden = x.narrow(-1, left_pad_, valid_length).contiguous();
    return hidden;
  }

 private:
  int64_t left_pad_;
  int64_t right_pad_;
  bool conv_weight_loaded_ = false;
  bool conv_bias_loaded_ = false;
  torch::nn::ConvTranspose1d conv_{nullptr};
};
TORCH_MODULE(Qwen3OmniCausalTransConvNext);

class Qwen3OmniCausalConvNextImpl : public torch::nn::Module {
 public:
  Qwen3OmniCausalConvNextImpl(const ModelContext& context,
                              int in_channels,
                              int out_channels,
                              int kernel_size,
                              int dilations = 1,
                              int groups = 1,
                              int stride = 1) {
    dwconv_ = torch::nn::Conv1d(
        torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(0)
            .dilation(dilations)
            .groups(groups));
    stride_ = stride;
    kernel_size_ = (kernel_size - 1) * dilations + 1;
    padding_ = kernel_size_ - stride_;
    register_module("dwconv", dwconv_);
  }

  void load_state_dict(const StateDict& state_dict) {
    auto dwconv_weight = state_dict.get_tensor("weight");
    if (dwconv_weight.defined()) {
      dwconv_->weight.data().copy_(dwconv_weight);
      dwconv_weight_loaded_ = true;
    }

    auto dwconv_bias = state_dict.get_tensor("bias");
    if (dwconv_bias.defined()) {
      dwconv_->bias.data().copy_(dwconv_bias);
      dwconv_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(dwconv_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(dwconv_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv.bias";
  }

  int64_t get_extra_padding_for_conv1d(torch::Tensor x) {
    int64_t length = x.size(-1);
    double n_frames =
        static_cast<double>(length - kernel_size_ + padding_) / stride_ + 1.0;
    int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride_ +
        (kernel_size_ - padding_);
    return ideal_length - length;
  }

  torch::Tensor forward(torch::Tensor x) {
    auto extra_padding = get_extra_padding_for_conv1d(x);
    x = torch::nn::functional::pad(
        x,
        torch::nn::functional::PadFuncOptions({padding_, extra_padding})
            .mode(torch::kConstant)
            .value(0));
    auto out = dwconv_->forward(x);
    return out.contiguous();
  }

 private:
  int kernel_size_;
  int stride_;
  int padding_;
  torch::nn::Conv1d dwconv_{nullptr};
  bool dwconv_weight_loaded_;
  bool dwconv_bias_loaded_;
};
TORCH_MODULE(Qwen3OmniCausalConvNext);

class Qwen3OmniConvNextBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniConvNextBlockImpl(const ModelContext& context, int dim) {
    dwconv_ = register_module(
        "dwconv", Qwen3OmniCausalConvNext(context, dim, dim, 7, 1, dim));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}).eps(1e-6)));
    pwconv1_ = register_module("pwconv1_", torch::nn::Linear(dim, 4 * dim));
    pwconv2_ = register_module("pwconv2_", torch::nn::Linear(4 * dim, dim));
    act_ = register_module("act", torch::nn::GELU());
    gamma_ = register_parameter(
        "gamma", torch::full({dim}, 1e-6, torch::requires_grad(true)));
  }
  void load_state_dict(const StateDict& state_dict) {
    dwconv_->load_state_dict(state_dict.get_dict_with_prefix("dwconv.conv."));

    auto norm_weight = state_dict.get_tensor("norm.weight");
    if (norm_weight.defined()) {
      norm_->weight.data().copy_(norm_weight);
      norm_weight_loaded_ = true;
    }
    auto norm_bias = state_dict.get_tensor("norm.bias");
    if (norm_bias.defined()) {
      norm_->bias.data().copy_(norm_bias);
      norm_bias_loaded_ = true;
    }

    auto pwconv1_weight = state_dict.get_tensor("pwconv1.weight");
    if (pwconv1_weight.defined()) {
      pwconv1_->weight.data().copy_(pwconv1_weight);
      pwconv1_weight_loaded_ = true;
    }
    auto pwconv1_bias = state_dict.get_tensor("pwconv1.bias");
    if (pwconv1_bias.defined()) {
      pwconv1_->bias.data().copy_(pwconv1_bias);
      pwconv1_bias_loaded_ = true;
    }

    auto pwconv2_weight = state_dict.get_tensor("pwconv2.weight");
    if (pwconv2_weight.defined()) {
      pwconv2_->weight.data().copy_(pwconv2_weight);
      pwconv2_weight_loaded_ = true;
    }
    auto pwconv2_bias = state_dict.get_tensor("pwconv2.bias");
    if (pwconv2_bias.defined()) {
      pwconv2_->bias.data().copy_(pwconv2_bias);
      pwconv2_bias_loaded_ = true;
    }

    auto gamma_weight = state_dict.get_tensor("gamma");
    if (gamma_weight.defined()) {
      gamma_.data().copy_(gamma_weight);
      gamma_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm.weight";
    CHECK(norm_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm.bias";
    CHECK(pwconv1_weight_loaded_)
        << "weight is not loaded for " << prefix + "pwconv1.weight";
    CHECK(pwconv1_bias_loaded_)
        << "bias is not loaded for " << prefix + "pwconv1.bias";
    CHECK(pwconv2_weight_loaded_)
        << "weight is not loaded for " << prefix + "pwconv2.weight";
    CHECK(pwconv2_bias_loaded_)
        << "bias is not loaded for " << prefix + "pwconv2.bias";
    CHECK(gamma_weight_loaded_)
        << "weight is not loaded for " << prefix + "gamma";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto input = x;
    x = dwconv_->forward(x);
    x = x.permute({0, 2, 1});
    x = norm_->forward(x);
    x = pwconv1_->forward(x);
    x = act_->forward(x);
    x = pwconv2_->forward(x);
    x = gamma_ * x;
    x = x.permute({0, 2, 1});
    x = x + input;
    return x;
  }

 private:
  Qwen3OmniCausalConvNext dwconv_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear pwconv1_{nullptr};
  torch::nn::Linear pwconv2_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::Tensor gamma_;
  bool norm_weight_loaded_ = false;
  bool norm_bias_loaded_ = false;
  bool pwconv1_weight_loaded_ = false;
  bool pwconv1_bias_loaded_ = false;
  bool pwconv2_weight_loaded_ = false;
  bool pwconv2_bias_loaded_ = false;
  bool gamma_weight_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniConvNextBlock);

class Qwen3OmniCode2WavSnakeBetaImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavSnakeBetaImpl(const ModelContext& context,
                                 int64_t in_features,
                                 float alpha = 1.0) {
    alpha_ = register_parameter(
        "alpha",
        torch::full(
            {in_features}, std::log(alpha), torch::requires_grad(true)));
    beta_ = register_parameter(
        "beta", torch::full({in_features}, 0.0, torch::requires_grad(true)));
  }
  void load_state_dict(const StateDict& state_dict) {
    auto alpha_weight = state_dict.get_tensor("alpha");
    if (alpha_weight.defined()) {
      alpha_.data().copy_(alpha_weight);
      alpha_weight_loaded_ = true;
    }

    auto beta_weight = state_dict.get_tensor("beta");
    if (beta_weight.defined()) {
      beta_.data().copy_(beta_weight);
      beta_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(alpha_weight_loaded_)
        << "weight is not loaded for " << prefix + "alpha";
    CHECK(beta_weight_loaded_)
        << "weight is not loaded for " << prefix + "beta";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto alpha_expanded = alpha_.view({1, -1, 1});
    auto beta_expanded = beta_.view({1, -1, 1});
    auto alpha_pos = torch::exp(alpha_expanded);
    auto beta_pos = torch::exp(beta_expanded);
    auto sin_sq = torch::pow(torch::sin(x * alpha_pos), 2);
    auto inv_beta = 1.0 / (beta_pos + no_div_by_zero);
    auto output = x + inv_beta * sin_sq;
    return output;
  }

 private:
  double no_div_by_zero = 1e-9;
  torch::Tensor alpha_;
  torch::Tensor beta_;
  bool alpha_weight_loaded_ = false;
  bool beta_weight_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniCode2WavSnakeBeta);

class Qwen3OmniCode2WavRmsNormImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavRmsNormImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    variance_epsilon_ = 1e-6;
    norm_ = register_parameter("norm", torch::ones(hidden_size));
  }
  void load_state_dict(const StateDict& state_dict) {
    auto norm_weight = state_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      norm_.data().copy_(norm_weight);
      norm_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto input_dtype = x.dtype();
    x = x.to(torch::kFloat32);
    auto variance = torch::mean(torch::pow(x, 2), -1, true).to(device_);
    x = x * torch::rsqrt(variance + variance_epsilon_);
    return norm_.to(device_) * x.to(input_dtype);
  }

 private:
  bool norm_weight_loaded_ = false;
  torch::Tensor norm_;
  double variance_epsilon_;
  torch::Device device_;
};
TORCH_MODULE(Qwen3OmniCode2WavRmsNorm);

class Qwen3OmniCode2WavRoteryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavRoteryEmbeddingImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto model_args = context.get_model_args();
    auto base = model_args.code2wav_config_rope_theta();
    auto partial_rotary_factory = 1.0;
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto num_attention_heads = model_args.code2wav_config_num_attention_heads();
    auto head_dim = hidden_size / num_attention_heads;
    auto dim = int(head_dim * partial_rotary_factory);
    attention_scaling_ = 1.0;
    dtype_ = context.get_tensor_options().dtype().toScalarType();
    auto indices =
        torch::arange(0, dim, 2, torch::TensorOptions().dtype(torch::kInt64));
    indices = indices.to(context.get_tensor_options().device());
    auto indices_f = indices.to(torch::kFloat32);
    auto freqs = indices_f / static_cast<float>(dim);
    auto base_tensor = torch::full_like(freqs, static_cast<float>(base));
    auto base_pow = torch::pow(base_tensor, freqs);
    inv_freq_ = base_pow.reciprocal();
    register_buffer("inv_freq", inv_freq_);
  }
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights(const std::string& prefix) const {}

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor position_ids) {
    auto inv_freq_expanded = inv_freq_.to(torch::kFloat32)
                                 .view({1, -1, 1})
                                 .expand({position_ids.size(0), -1, 1})
                                 .to(device_)
                                 .to(torch::kFloat32);
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);

    auto freqs =
        torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
    auto emb = torch::cat({freqs, freqs}, -1);
    auto cos = torch::cos(emb) * attention_scaling_;
    auto sin = torch::sin(emb) * attention_scaling_;
    return std::make_pair(cos.to(dtype_), sin.to(dtype_));
  }

 private:
  torch::Tensor inv_freq_;
  torch::Device device_;
  torch::ScalarType dtype_;
  float attention_scaling_;
};
TORCH_MODULE(Qwen3OmniCode2WavRoteryEmbedding);

class Qwen3OmniCode2WavPretransformerImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavPretransformerImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto model_args = context.get_model_args();
    auto num_hidden_layers = model_args.code2wav_config_num_hidden_layers();
    sliding_window_ = model_args.code2wav_config_sliding_window();
    blocks_ = register_module("transformer_layer", torch::nn::ModuleList());
    layers_.reserve(model_args.code2wav_config_num_hidden_layers());
    has_sliding_layers_ = true;
    options_ = context.get_tensor_options();
    rms_norm_ = register_module("norm", Qwen3OmniCode2WavRmsNorm(context));
    rotery_embedding_ = register_module(
        "rotery_embedding", Qwen3OmniCode2WavRoteryEmbedding(context));
    for (int32_t i = 0; i < model_args.code2wav_config_num_hidden_layers();
         i++) {
      auto block = layer::Qwen3OmniCode2WavTransformerLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    rms_norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    rms_norm_->verify_loaded_weights(prefix + "norm.");
  }

  torch::Tensor forward(torch::Tensor x, ModelInputParams& input_params) {
    auto pasr_seen_tokens = 0;
    auto seq_len = x.size(1);
    auto cache_position = torch::arange(
        pasr_seen_tokens,
        pasr_seen_tokens + seq_len,
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto position_ids = cache_position.unsqueeze(0);
    auto attention_mask = _get_sliding_attention_mask(
        x, sliding_window_, cache_position, position_ids);

    auto sliding_window_size = sliding_window_;
    auto position_embeddings = rotery_embedding_->forward(x, position_ids);
    auto m_cos = position_embeddings.first;
    auto m_sin = position_embeddings.second;
    auto cos_flat =
        m_cos.reshape({m_cos.size(0) * m_cos.size(1), m_cos.size(2)})
            .to(device_);
    auto sin_flat =
        m_sin.reshape({m_sin.size(0) * m_sin.size(1), m_sin.size(2)})
            .to(device_);
    auto x_bs = x.size(0);
    auto x_token = x.size(1);
    auto x_flat = x.reshape({x_bs * x_token, x.size(2)}).to(device_);

    std::vector<int32_t> cu_seqlens_vec;
    cu_seqlens_vec.reserve(x_bs);
    for (int64_t i = 0; i < x_bs; ++i) {
      cu_seqlens_vec.push_back(static_cast<int32_t>((i + 1) * x_token));
    }
    auto cu_seqlen = torch::tensor(
        cu_seqlens_vec,
        torch::TensorOptions().dtype(torch::kInt32).device(device_));

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      auto& layer = layers_[i];
      layer(x_flat,
            cos_flat,
            sin_flat,
            attention_mask,
            cu_seqlen,
            cu_seqlens_vec,
            input_params,
            i,
            event,
            event_flag);
    }
    x = rms_norm_->forward(x);
    return x;
  }

  torch::Tensor _get_min_value(const at::TensorOptions& options) {
    return torch::full({}, 1.0, options);
  }

  torch::Tensor _get_sliding_attention_mask(torch::Tensor hidden_states,
                                            int window_size,
                                            torch::Tensor cache_position,
                                            torch::Tensor position_ids) {
    int64_t query_len = hidden_states.size(1);
    int64_t key_len = query_len;

    torch::Tensor pos = position_ids.select(0, 0);
    torch::Tensor query_pos = pos.slice(0, 0, query_len).unsqueeze(-1);
    torch::Tensor key_pos = pos.unsqueeze(0);
    torch::Tensor distance = (query_pos - key_pos).abs();
    torch::Tensor valid_mask =
        (distance < window_size) & (key_pos <= query_pos);

    auto opts = options_;
    torch::Tensor attention_mask = torch::zeros({query_len, key_len}, opts);
    torch::Tensor min_val = _get_min_value(opts);
    attention_mask = torch::where(valid_mask, attention_mask, min_val);

    return attention_mask;
  }

 private:
  bool has_sliding_layers_;
  int64_t sliding_window_;
  torch::Device device_;
  torch::TensorOptions options_;
  torch::nn::ModuleList blocks_ = nullptr;
  Qwen3OmniCode2WavRmsNorm rms_norm_{nullptr};
  Qwen3OmniCode2WavRoteryEmbedding rotery_embedding_{nullptr};
  std::vector<layer::Qwen3OmniCode2WavTransformerLayer> layers_;
};
TORCH_MODULE(Qwen3OmniCode2WavPretransformer);

class Qwen3OmniCode2WavUnsampleImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavUnsampleImpl(const ModelContext& context) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto upsampling_ratios_vec =
        model_args.code2wav_config_upsampling_ratios_vec();
    upsample_layers_ =
        register_module("upsample_layers", torch::nn::ModuleList());
    causalTransConvNext_layers_.reserve(upsampling_ratios_vec.size());
    convNextBlock_layers_.reserve(upsampling_ratios_vec.size());
    for (size_t i = 0; i < upsampling_ratios_vec.size(); i++) {
      int64_t factor = upsampling_ratios_vec[i];
      auto block_1 = Qwen3OmniCausalTransConvNext(
          context, hidden_size, hidden_size, factor, factor);
      auto block_2 = Qwen3OmniConvNextBlock(context, hidden_size);
      upsample_layers_->push_back(block_1);
      causalTransConvNext_layers_.push_back(block_1);
      upsample_layers_->push_back(block_2);
      convNextBlock_layers_.push_back(block_2);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      x = causalTransConvNext_layers_[i]->forward(x);
      x = convNextBlock_layers_[i]->forward(x);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      causalTransConvNext_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i) + ".0.conv."));
      convNextBlock_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i) + ".1."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      causalTransConvNext_layers_[i]->verify_loaded_weights(
          prefix + std::to_string(i) + ".0.conv.");
      convNextBlock_layers_[i]->verify_loaded_weights(
          prefix + std::to_string(i) + ".1.");
    }
  }

 private:
  torch::nn::ModuleList upsample_layers_{nullptr};
  std::vector<Qwen3OmniCausalTransConvNext> causalTransConvNext_layers_;
  std::vector<Qwen3OmniConvNextBlock> convNextBlock_layers_;
};
TORCH_MODULE(Qwen3OmniCode2WavUnsample);

class Qwen3OmniCode2WavDecoderResidualUnitImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderResidualUnitImpl(const ModelContext& context,
                                           int dim = 16,
                                           int dilation = 1) {
    act1_ = register_module("act1", Qwen3OmniCode2WavSnakeBeta(context, dim));
    conv1_ = register_module(
        "conv1", Qwen3OmniCausalConvNext(context, dim, dim, 7, dilation));
    act2_ = register_module("act2", Qwen3OmniCode2WavSnakeBeta(context, dim));
    conv2_ =
        register_module("conv2", Qwen3OmniCausalConvNext(context, dim, dim, 1));
  }

  void load_state_dict(const StateDict& state_dict) {
    act1_->load_state_dict(state_dict.get_dict_with_prefix("act1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1.conv."));
    act2_->load_state_dict(state_dict.get_dict_with_prefix("act2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2.conv."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    act1_->verify_loaded_weights(prefix + "act1.");
    conv1_->verify_loaded_weights(prefix + "conv1.conv.");
    act2_->verify_loaded_weights(prefix + "act2.");
    conv2_->verify_loaded_weights(prefix + "conv2.conv.");
  }

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;
    x = act1_->forward(x);
    x = conv1_->forward(x);
    x = act2_->forward(x);
    x = conv2_->forward(x);
    return x + residual;
  }

 private:
  Qwen3OmniCode2WavSnakeBeta act1_{nullptr}, act2_{nullptr};
  Qwen3OmniCausalConvNext conv1_{nullptr}, conv2_{nullptr};
};
TORCH_MODULE(Qwen3OmniCode2WavDecoderResidualUnit);

class Qwen3OmniCode2WavDecoderBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderBlockImpl(const ModelContext& context,
                                    int layer_idx) {
    auto model_args = context.get_model_args();
    auto decoder_dim = model_args.code2wav_config_decoder_dim();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();
    auto upsample_rate = upsample_rates_vec[layer_idx];
    auto in_dim = decoder_dim >> layer_idx;
    auto out_dim = decoder_dim >> (layer_idx + 1);
    snk_ = register_module("snk", Qwen3OmniCode2WavSnakeBeta(context, in_dim));
    trans_conv_ = register_module(
        "trans_conv",
        Qwen3OmniCausalTransConvNext(
            context, in_dim, out_dim, 2 * upsample_rate, upsample_rate));

    residual_module_ =
        register_module("residual_items", torch::nn::ModuleList());
    residual_items_.reserve(3);

    for (int32_t i = 0; i < 3; i++) {
      auto item =
          Qwen3OmniCode2WavDecoderResidualUnit(context, out_dim, pow(3, i));
      residual_module_->push_back(item);
      residual_items_.push_back(item);
    }
  }
  void load_state_dict(const StateDict& state_dict) {
    snk_->load_state_dict(state_dict.get_dict_with_prefix("0."));
    trans_conv_->load_state_dict(state_dict.get_dict_with_prefix("1.conv."));

    for (int32_t i = 0; i < 3; i++) {
      residual_items_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i + 2) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    snk_->verify_loaded_weights(prefix + "0.");
    trans_conv_->verify_loaded_weights(prefix + "1.conv.");

    for (int32_t i = 0; i < 3; i++) {
      residual_items_[i]->verify_loaded_weights(prefix + std::to_string(i + 2) +
                                                ".");
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    x = snk_->forward(x);
    x = trans_conv_->forward(x);

    for (int32_t i = 0; i < 3; i++) {
      x = residual_items_[i]->forward(x);
    }
    return x;
  }

 private:
  Qwen3OmniCode2WavSnakeBeta snk_{nullptr};
  Qwen3OmniCausalTransConvNext trans_conv_{nullptr};
  torch::nn::ModuleList decoder_block_{nullptr};
  torch::nn::ModuleList residual_module_{nullptr};
  std::vector<Qwen3OmniCode2WavDecoderResidualUnit> residual_items_;
};
TORCH_MODULE(Qwen3OmniCode2WavDecoderBlock);

class Qwen3OmniCode2WavDecoderImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto decoder_dim = model_args.code2wav_config_decoder_dim();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();

    int output_dim = decoder_dim >> upsample_rates_vec.size();
    casual_conv_1_ = register_module(
        "casual_conv_1",
        Qwen3OmniCausalConvNext(context, hidden_size, decoder_dim, 7));
    snk_beta_ = register_module(
        "snk_beta", Qwen3OmniCode2WavSnakeBeta(context, output_dim));
    casual_conv_2_ = register_module(
        "casual_conv_2", Qwen3OmniCausalConvNext(context, output_dim, 1, 7));
    blocks_module_ = register_module("decoder_layer", torch::nn::ModuleList());
    decoder_blocks_.reserve(4);

    for (int32_t i = 0; i < 4; i++) {
      auto block = Qwen3OmniCode2WavDecoderBlock(context, i);
      blocks_module_->push_back(block);
      decoder_blocks_.push_back(block);
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    casual_conv_1_->load_state_dict(state_dict.get_dict_with_prefix("0.conv."));

    for (int i = 0; i < decoder_blocks_.size(); i++) {
      decoder_blocks_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i + 1) + ".block."));
    }

    snk_beta_->load_state_dict(state_dict.get_dict_with_prefix("5."));
    casual_conv_2_->load_state_dict(state_dict.get_dict_with_prefix("6.conv."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    casual_conv_1_->verify_loaded_weights(prefix + "0.conv.");

    for (int i = 0; i < decoder_blocks_.size(); i++) {
      decoder_blocks_[i]->verify_loaded_weights(prefix + std::to_string(i + 1) +
                                                ".block.");
    }

    snk_beta_->verify_loaded_weights(prefix + "5.");
    casual_conv_2_->verify_loaded_weights(prefix + "6.conv.");
  }

  torch::Tensor forward(torch::Tensor x) {
    x = casual_conv_1_->forward(x);

    for (int i = 0; i < decoder_blocks_.size(); i++) {
      x = decoder_blocks_[i]->forward(x);
    }

    x = snk_beta_->forward(x);
    x = casual_conv_2_->forward(x);
    return x;
  }

 private:
  torch::nn::ModuleList blocks_module_{nullptr};
  std::vector<Qwen3OmniCode2WavDecoderBlock> decoder_blocks_;
  Qwen3OmniCausalConvNext casual_conv_1_{nullptr}, casual_conv_2_{nullptr};
  Qwen3OmniCode2WavSnakeBeta snk_beta_{nullptr};
};

TORCH_MODULE(Qwen3OmniCode2WavDecoder);

class Qwen3OmniCode2WavImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    options_ = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto codebook_size = model_args.code2wav_config_codebook_size();
    auto num_quantizers = model_args.code2wav_config_num_quantizers();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto upsampling_ratios_vec =
        model_args.code2wav_config_upsampling_ratios_vec();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();

    total_upsample_ = 1;
    for (int64_t ratio : upsampling_ratios_vec) {
      total_upsample_ *= ratio;
    }
    for (int64_t rate : upsample_rates_vec) {
      total_upsample_ *= rate;
    }

    code_embedding_ = register_module(
        "code_embedding",
        torch::nn::Embedding(codebook_size * num_quantizers, hidden_size));
    code_embedding_->weight.set_data(code_embedding_->weight.to(options_));

    code_offset_ = (torch::arange(num_quantizers, torch::kLong) * codebook_size)
                       .view({1, -1, 1})
                       .to(options_);
    code_offset_ = code_offset_.to(torch::kLong);

    pretransformer_ = register_module("pre_transformer",
                                      Qwen3OmniCode2WavPretransformer(context));

    unsample_layer_ =
        register_module("unsample", Qwen3OmniCode2WavUnsample(context));
    unsample_layer_->to(device_);

    decoder_layer_ =
        register_module("decoder", Qwen3OmniCode2WavDecoder(context));
    decoder_layer_->to(device_);
  }
  torch::Tensor forward(const torch::Tensor& input,
                        ModelInputParams& input_params) {
    auto x = input;
    auto state_dict = StateDictFromSafeTensor::load(
        "/export/home/weinan5/zhubowei/github_code/xllm/"
        "code2wav.safetensors");
    auto input_featss = torch::empty({1, 16, 149}, torch::kLong);
    bool is_conv_out_weight_loaded_ = false;
    weight::load_weight(
        *state_dict, "codes", input_featss, is_conv_out_weight_loaded_);
    x = input_featss.to(device_).to(torch::kLong);

    torch::Tensor hidden = code_embedding_(x + code_offset_).mean(1);
    hidden = pretransformer_->forward(hidden, input_params);
    hidden = hidden.permute({0, 2, 1});
    hidden = unsample_layer_->forward(hidden);
    hidden = decoder_layer_->forward(hidden);
    torch::Tensor decoder_output = hidden.to(torch::kCPU).to(torch::kFloat32);
    torch::save(decoder_output, "decoder_output_cpp.pt");
    LOG(INFO) << "end do the code devoder";
    return torch::clamp(hidden, -1.0, 1.0);
  }

  void load_model(std::shared_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto emb_weight =
          state_dict->get_tensor("code2wav.code_embedding.weight");
      if (emb_weight.defined()) {
        code_embedding_->weight.data().copy_(emb_weight);
        code_embedding_weight_loaded_ = true;
      }
      pretransformer_->load_state_dict(
          state_dict->get_dict_with_prefix("code2wav.pre_transformer."));
      pretransformer_->merge_loaded_weights();
      unsample_layer_->load_state_dict(
          state_dict->get_dict_with_prefix("code2wav.upsample."));
      decoder_layer_->load_state_dict(
          state_dict->get_dict_with_prefix("code2wav.decoder."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(code_embedding_weight_loaded_)
        << "weight is not loaded for " << prefix + "code_embedding.weight";
    pretransformer_->verify_loaded_weights(prefix +
                                           "code2wav.pre_transformer.");
    unsample_layer_->verify_loaded_weights(prefix + "code2wav.upsample.");
    decoder_layer_->verify_loaded_weights(prefix + "code2wav.decoder.");
  }

  torch::Tensor chunked_decode(const torch::Tensor& codes,
                               ModelInputParams& input_params,
                               int64_t chunk_size = 300,
                               int64_t left_context_size = 25) {
    TORCH_CHECK(codes.dim() >= 2, "codes must have at least 2 dim");
    int64_t total_length = codes.size(-1);
    std::vector<torch::Tensor> wavs;
    int64_t start_index = 0;

    while (start_index < total_length) {
      int64_t end_index = std::min(start_index + chunk_size, total_length);
      int64_t context_start = (start_index >= left_context_size)
                                  ? start_index - left_context_size
                                  : 0;
      int64_t context_size = start_index - context_start;

      std::vector<torch::indexing::TensorIndex> slice_indices;
      for (int64_t i = 0; i < codes.dim() - 1; ++i) {
        slice_indices.push_back(torch::indexing::Slice());
      }
      slice_indices.push_back(torch::indexing::Slice(context_start, end_index));
      auto codes_chunk = codes.index(slice_indices);
      auto wav_chunk = forward(codes_chunk, input_params);
      int64_t output_context_samples = context_size * total_upsample_;
      std::vector<torch::indexing::TensorIndex> output_slice;
      for (int64_t i = 0; i < wav_chunk.dim() - 1; ++i) {
        output_slice.push_back(torch::indexing::Slice());
      }
      output_slice.push_back(
          torch::indexing::Slice(output_context_samples, None));
      auto wav_cropped = wav_chunk.index(output_slice);

      wavs.push_back(wav_cropped);
      start_index = end_index;
    }
    return torch::cat(wavs, -1);
  }

 private:
  torch::Tensor code_offset_;
  int64_t total_upsample_;
  torch::TensorOptions options_;
  torch::Device device_;
  bool code_embedding_weight_loaded_ = false;
  torch::nn::Embedding code_embedding_{nullptr};
  Qwen3OmniCode2WavPretransformer pretransformer_{nullptr};
  Qwen3OmniCode2WavUnsample unsample_layer_{nullptr};
  Qwen3OmniCode2WavDecoder decoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3OmniCode2Wav);
}  // namespace xllm
