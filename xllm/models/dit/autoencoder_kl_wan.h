#pragma once
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "autoencoder_kl.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"

namespace xllm {

class AvgDown3DImpl : public torch::nn::Module {
 public:
  AvgDown3DImpl(int64_t in_channels,
                int64_t out_channels,
                int64_t factor_t,
                int64_t factor_s = 1)
      : in_channels_(in_channels),
        out_channels_(out_channels),
        factor_t_(factor_t),
        factor_s_(factor_s) {
    factor_ = factor_t_ * factor_s_ * factor_s_;
    TORCH_CHECK(in_channels_ * factor_ % out_channels_ == 0,
                "in_channels * factor must be divisible by out_channels");
    group_size_ = in_channels_ * factor_ / out_channels_;
  }

  torch::Tensor forward(torch::Tensor x) {
    int64_t pad_t = (factor_t_ - x.size(2) % factor_t_) % factor_t_;
    std::vector<int64_t> pad = {0, 0, 0, 0, pad_t, 0};
    x = torch::nn::functional::pad(x,
                                   torch::nn::functional::PadFuncOptions(pad));
    auto sizes = x.sizes();
    int64_t B = sizes[0], C = sizes[1], T = sizes[2], H = sizes[3],
            W = sizes[4];
    x = x.view({B,
                C,
                T / factor_t_,
                factor_t_,
                H / factor_s_,
                factor_s_,
                W / factor_s_,
                factor_s_});
    x = x.permute({0, 1, 3, 5, 7, 2, 4, 6}).contiguous();
    x = x.view({B, C * factor_, T / factor_t_, H / factor_s_, W / factor_s_});
    x = x.view({B,
                out_channels_,
                group_size_,
                T / factor_t_,
                H / factor_s_,
                W / factor_s_});
    x = x.mean(2);
    return x;
  }

 private:
  int64_t in_channels_, out_channels_, factor_t_, factor_s_, factor_,
      group_size_;
};
TORCH_MODULE(AvgDown3D);

class DupUp3DImpl : public torch::nn::Module {
 public:
  DupUp3DImpl(int64_t in_channels,
              int64_t out_channels,
              int64_t factor_t,
              int64_t factor_s = 1)
      : in_channels_(in_channels),
        out_channels_(out_channels),
        factor_t_(factor_t),
        factor_s_(factor_s) {
    factor_ = factor_t_ * factor_s_ * factor_s_;
    TORCH_CHECK(out_channels_ * factor_ % in_channels_ == 0,
                "out_channels * factor must be divisible by in_channels");
    repeats_ = out_channels_ * factor_ / in_channels_;
  }

  torch::Tensor forward(torch::Tensor x, bool first_chunk = false) {
    x = x.repeat_interleave(repeats_, 1);
    x = x.view({x.size(0),
                out_channels_,
                factor_t_,
                factor_s_,
                factor_s_,
                x.size(2),
                x.size(3),
                x.size(4)});
    x = x.permute({0, 1, 5, 2, 6, 3, 7, 4}).contiguous();
    x = x.view({x.size(0),
                out_channels_,
                x.size(2) * factor_t_,
                x.size(4) * factor_s_,
                x.size(6) * factor_s_});
    if (first_chunk) {
      x = x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(factor_t_ - 1, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()});
    }
    return x;
  }

 private:
  int64_t in_channels_, out_channels_, factor_t_, factor_s_, factor_, repeats_;
};
TORCH_MODULE(DupUp3D);

class WanCausalConv3DImpl : public torch::nn::Module {
 public:
  WanCausalConv3DImpl(int64_t in_channels,
                      int64_t out_channels,
                      std::vector<int64_t> kernel_size,
                      std::vector<int64_t> stride = {1, 1, 1},
                      std::vector<int64_t> padding = {0, 0, 0})
      : in_channels_(in_channels),
        out_channels_(out_channels),
        kernel_size_(kernel_size),
        stride_(stride),
        padding_(padding) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(0)
                .bias(true)));
    _padding_ = {
        padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0};
  }

  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::optional<torch::Tensor>& cache_x = torch::nullopt) {
    std::vector<int64_t> padding = _padding_;
    torch::Tensor input = x;
    if (cache_x.has_value() && padding[4] > 0) {
      torch::Tensor cache = cache_x.value().to(x.device());
      input = torch::cat({cache, input}, 2);
      padding[4] -= cache.size(2);
    }
    input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(padding));
    return conv_->forward(input);
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("conv.weight");
    if (weight.defined()) {
      DCHECK_EQ(conv_->weight.sizes(), weight.sizes())
          << "conv weight size mismatch";
      conv_->weight.data().copy_(weight);
    }
    const auto bias = state_dict.get_tensor("conv.bias");
    if (bias.defined() && conv_->bias.defined()) {
      DCHECK_EQ(conv_->bias.sizes(), bias.sizes()) << "conv bias size mismatch";
      conv_->bias.data().copy_(bias);
    }
  }

 private:
  int64_t in_channels_, out_channels_;
  std::vector<int64_t> kernel_size_, stride_, padding_, _padding_;
  torch::nn::Conv3d conv_{nullptr};
};
TORCH_MODULE(WanCausalConv3D);

class WanRMSNormImpl : public torch::nn::Module {
 public:
  WanRMSNormImpl(int64_t dim,
                 bool channel_first = true,
                 bool images = true,
                 bool bias = false)
      : channel_first_(channel_first), images_(images), bias_enabled_(bias) {
    std::vector<int64_t> broadcastable_dims;
    if (!images) {
      broadcastable_dims = {1, 1, 1};
    } else {
      broadcastable_dims = {1, 1};
    }
    std::vector<int64_t> shape;
    if (channel_first) {
      shape.push_back(dim);
      shape.insert(
          shape.end(), broadcastable_dims.begin(), broadcastable_dims.end());
    } else {
      shape.push_back(dim);
    }
    scale_ = std::sqrt(static_cast<double>(dim));
    gamma_ = register_parameter("gamma", torch::ones(shape));
    if (bias_enabled_) {
      bias_ = register_parameter("bias", torch::zeros(shape));
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    int64_t norm_dim = channel_first_ ? 1 : -1;
    auto normed = torch::nn::functional::normalize(
        x, torch::nn::functional::NormalizeFuncOptions().dim(norm_dim));
    auto out = normed * scale_ * gamma_;
    if (bias_enabled_) {
      out = out + bias_;
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto gamma_weight = state_dict.get_tensor("gamma");
    if (gamma_weight.defined()) {
      gamma_.data().copy_(gamma_weight);
    }
    if (bias_enabled_) {
      const auto bias_weight = state_dict.get_tensor("bias");
      if (bias_weight.defined()) {
        bias_.data().copy_(bias_weight);
      }
    }
  }

 private:
  bool channel_first_;
  bool images_;
  bool bias_enabled_;
  double scale_;
  torch::Tensor gamma_;
  torch::Tensor bias_;
};
TORCH_MODULE(WanRMSNorm);

class WanResampleImpl : public torch::nn::Module {
 public:
  WanResampleImpl(int64_t dim,
                  const std::string& mode,
                  int64_t upsample_out_dim = -1)
      : dim_(dim), mode_(mode) {
    if (upsample_out_dim == -1) {
      upsample_out_dim = dim / 2;
    }

    if (mode == "upsample2d") {
      auto resample = torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{2.0, 2.0})
                                  .mode("nearest")),
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(dim, upsample_out_dim, 3).padding(1)));
    } else if (mode == "upsample3d") {
      auto resample = torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{2.0, 2.0})
                                  .mode("nearest")),
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(dim, upsample_out_dim, 3).padding(1)));
      time_conv_ =
          register_module("time_conv",
                          WanCausalConv3D(dim,
                                          dim,
                                          std::vector<int64_t>{3, 1, 1},
                                          std::vector<int64_t>{1, 1, 1},
                                          std::vector<int64_t>{1, 0, 0}));
    } else if (mode == "downsample2d") {
      resample = torch::nn::Sequential(
          torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)
                                .stride(std::vector<int64_t>{2, 2})));
    } else if (mode == "downsample3d") {
      auto resample = torch::nn::Sequential(
          torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)
                                .stride(std::vector<int64_t>{2, 2})));
      time_conv_ =
          register_module("time_conv",
                          WanCausalConv3D(dim,
                                          dim,
                                          std::vector<int64_t>{3, 1, 1},
                                          std::vector<int64_t>{2, 1, 1},
                                          std::vector<int64_t>{0, 0, 0}));
    } else {
      auto resample = torch::nn::Sequential(torch::nn::Identity());
    }
    resample_ = register_module("resample", resample);
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0}) {
    auto sizes = x.sizes();
    int64_t b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3],
            w = sizes[4];

    if (mode == "upsample3d" && feat_cache) {
      int idx = (*feat_idx)[0];
      if ((*feat_cache)[idx].numel() == 0) {
        (*feat_cache)[idx] = torch::full({1}, -1, x.options());  // Rep flag
        (*feat_idx)[0] += 1;
      } else {
        auto cache_x =
            x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(),
                     torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                     torch::indexing::Slice(),
                     torch::indexing::Slice()})
                .clone();
        if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0 &&
            (*feat_cache)[idx].item<int>() != -1) {
          cache_x = torch::cat({(*feat_cache)[idx]
                                    .index({torch::indexing::Slice(),
                                            torch::indexing::Slice(),
                                            -1,
                                            torch::indexing::Slice(),
                                            torch::indexing::Slice()})
                                    .unsqueeze(2)
                                    .to(cache_x.device()),
                                cache_x},
                               2);
        }
        if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0 &&
            (*feat_cache)[idx].item<int>() == -1) {
          cache_x = torch::cat(
              {torch::zeros_like(cache_x).to(cache_x.device()), cache_x}, 2);
        }
        if ((*feat_cache)[idx].item<int>() == -1) {
          x = time_conv->forward(x);
        } else {
          x = time_conv->forward(x, (*feat_cache)[idx]);
        }
        (*feat_cache)[idx] = cache_x;
        (*feat_idx)[0] += 1;

        x = x.view({b, 2, c, t, h, w});
        x = torch::stack({x.index({torch::indexing::Slice(),
                                   0,
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice()}),
                          x.index({torch::indexing::Slice(),
                                   1,
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice()})},
                         3);
        x = x.view({b, c, t * 2, h, w});
      }
    }
    t = x.size(2);
    x = x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});
    x = resample->forward(x);
    x = x.view({b, t, x.size(1), x.size(2), x.size(3)})
            .permute({0, 2, 1, 3, 4});

    if (mode == "downsample3d" && feat_cache) {
      int idx = (*feat_idx)[0];
      if ((*feat_cache)[idx].numel() == 0) {
        (*feat_cache)[idx] = x.clone();
        (*feat_idx)[0] += 1;
      } else {
        auto cache_x =
            x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(),
                     torch::indexing::Slice(-1, torch::indexing::None),
                     torch::indexing::Slice(),
                     torch::indexing::Slice()})
                .clone();
        x = time_conv->forward(
            torch::cat({(*feat_cache)[idx].index(
                            {torch::indexing::Slice(),
                             torch::indexing::Slice(),
                             torch::indexing::Slice(-1, torch::indexing::None),
                             torch::indexing::Slice(),
                             torch::indexing::Slice()}),
                        x},
                       2));
        (*feat_cache)[idx] = cache_x;
        (*feat_idx)[0] += 1;
      }
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resample->size(); ++i) {
      auto module = resample[i];
      if (auto conv =
              std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module)) {
        const auto weight =
            state_dict.get_tensor("resample." + std::to_string(i) + ".weight");
        if (weight.defined()) {
          DCHECK_EQ(conv->weight.sizes(), weight.sizes())
              << "resample conv weight size mismatch: expected "
              << conv->weight.sizes() << " but got " << weight.sizes();
          conv->weight.data().copy_(weight);
        }
        const auto bias =
            state_dict.get_tensor("resample." + std::to_string(i) + ".bias");
        if (bias.defined() && conv->bias.defined()) {
          DCHECK_EQ(conv->bias.sizes(), bias.sizes())
              << "resample conv bias size mismatch: expected "
              << conv->bias.sizes() << " but got " << bias.sizes();
          conv->bias.data().copy_(bias);
        }
      }
    }
    if (time_conv) {
      time_conv->load_state_dict(state_dict.get_dict_with_prefix("time_conv."));
    }
  }

 private:
  int64_t dim_;
  std::string mode_;
  torch::nn::Sequential resample_{nullptr};
  WanCausalConv3D time_conv_{nullptr};
  const int CACHE_T = 2;
};
TORCH_MODULE(WanResample);

class WanResidualBlockImpl : public torch::nn::Module {
 public:
  WanResidualBlockImpl(int64_t in_dim, int64_t out_dim, float dropout = 0.0f)
      : in_dim_(in_dim), out_dim_(out_dim) {
    nonlinearity_ = torch::nn::Functional(torch::silu);
    norm1_ = register_module("norm1", WanRMSNorm(in_dim, true, false, false));
    conv1_ = register_module("conv1",
                             WanCausalConv3D(dim,
                                             dim,
                                             std::vector<int64_t>{3, 3, 3},
                                             std::vector<int64_t>{1, 1, 1},
                                             std::vector<int64_t>{1, 1, 1}));
    norm2_ = register_module("norm2", WanRMSNorm(out_dim, true, false, false));
    dropout_layer_ = register_module("dropout", torch::nn::Dropout(dropout));
    conv2_ = register_module("conv2",
                             WanCausalConv3D(dim,
                                             dim,
                                             std::vector<int64_t>{3, 3, 3},
                                             std::vector<int64_t>{1, 1, 1},
                                             std::vector<int64_t>{1, 1, 1}));
    if in_dim_
      != out_dim_ {
        conv_shortcut_ =
            register_module("conv_shortcut",
                            WanCausalConv3D(dim,
                                            dim,
                                            std::vector<int64_t>{1, 1, 1},
                                            std::vector<int64_t>{1, 1, 1},
                                            std::vector<int64_t>{0, 0, 0}));
      }
    else {
      conv_shortcut_ = register_module("conv_shortcut", torch::nn::Identity());
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0}) {
    torch::Tensor h;
    h = conv_shortcut_->forward(x);
    x = norm1_->forward(x);
    x = nonlinearity_(x);

    if (feat_cache) {
      int idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv1_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv1->forward(x);
    }

    x = norm2_->forward(x);
    x = nonlinearity_(x);
    x = dropout_layer_->forward(x);

    if (feat_cache) {
      int idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv2_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv2_->forward(x);
    }

    return x + h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
    conv_shortcut_->load_state_dict(
        state_dict.get_dict_with_prefix("conv_shortcut."));
  }

 private:
  int64_t in_dim_, out_dim_;
  std::string non_linearity_;
  const int CACHE_T = 2;

  torch::nn::Functional nonlinearity_{nullptr};
  WanRMSNorm norm1_{nullptr}, norm2_{nullptr};
  WanCausalConv3D conv1_{nullptr}, conv2_{nullptr}, conv_shortcut_{nullptr};
  torch::nn::Dropout dropout_layer_{nullptr};
};
TORCH_MODULE(WanResidualBlock);

class WanAttentionBlockImpl : public torch::nn::Module {
 public:
  WanAttentionBlockImpl(int64_t dim) : dim_(dim) {
    norm_ = register_module("norm", WanRMSNorm(dim, true, true, false));
    to_qkv_ = register_module(
        "to_qkv", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim * 3, 1)));
    proj_ = register_module(
        "proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 1)));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor identity = x;
    auto sizes = x.sizes();
    int64_t batch_size = sizes[0];
    int64_t channels = sizes[1];
    int64_t time = sizes[2];
    int64_t height = sizes[3];
    int64_t width = sizes[4];

    x = x.permute({0, 2, 1, 3, 4})
            .reshape({batch_size * time, channels, height, width});
    x = norm_->forward(x);

    auto qkv = to_qkv_->forward(x);
    qkv = qkv.reshape({batch_size * time, 1, channels * 3, height * width});
    qkv = qkv.permute({0, 1, 3, 2}).contiguous();
    auto chunks = qkv.chunk(3, -1);
    torch::Tensor q = chunks[0];
    torch::Tensor k = chunks[1];
    torch::Tensor v = chunks[2];

    torch::Tensor attn_out =
        torch::nn::functional::scaled_dot_product_attention(q, k, v);

    attn_out = attn_out.squeeze(1).permute({0, 2, 1}).reshape(
        {batch_size * time, channels, height, width});
    attn_out = proj_->forward(attn_out);

    attn_out = attn_out.view({batch_size, time, channels, height, width})
                   .permute({0, 2, 1, 3, 4});
    return attn_out + identity;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    const auto to_qkv_weight = state_dict.get_tensor("to_qkv.weight");
    if (to_qkv_weight.defined()) {
      DCHECK_EQ(to_qkv_->weight.sizes(), to_qkv_weight.sizes())
          << "to_qkv weight size mismatch";
      to_qkv_->weight.data().copy_(to_qkv_weight);
    }
    const auto to_qkv_bias = state_dict.get_tensor("to_qkv.bias");
    if (to_qkv_bias.defined() && to_qkv_->bias.defined()) {
      DCHECK_EQ(to_qkv_->bias.sizes(), to_qkv_bias.sizes())
          << "to_qkv bias size mismatch";
      to_qkv_->bias.data().copy_(to_qkv_bias);
    }
    const auto proj_weight = state_dict.get_tensor("proj.weight");
    if (proj_weight.defined()) {
      DCHECK_EQ(proj_->weight.sizes(), proj_weight.sizes())
          << "proj weight size mismatch";
      proj_->weight.data().copy_(proj_weight);
    }
    const auto proj_bias = state_dict.get_tensor("proj.bias");
    if (proj_bias.defined() && proj_->bias.defined()) {
      DCHECK_EQ(proj_->bias.sizes(), proj_bias.sizes())
          << "proj bias size mismatch";
      proj_->bias.data().copy_(proj_bias);
    }
  }

 private:
  int64_t dim_;
  WanRMSNorm norm_{nullptr};
  torch::nn::Conv2d to_qkv_{nullptr};
  torch::nn::Conv2d proj_{nullptr};
};
TORCH_MODULE(WanAttentionBlock);

class WanMidBlockImpl : public torch::nn::Module {
 public:
  WanMidBlockImpl(int64_t dim, float dropout = 0.0f, int num_layers = 1)
      : dim_(dim) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    resnets_->push_back(WanResidualBlock(dim, dim, dropout));
    for (int i = 0; i < num_layers; ++i) {
      attentions_->push_back(WanAttentionBlock(dim));
      resnets_->push_back(WanResidualBlock(dim, dim, dropout));
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0}) {
    x = resnets_[0]->as<WanResidualBlock>()->forward(x, feat_cache, feat_idx);
    for (size_t i = 0; i < attentions_->size(); ++i) {
      auto attn = attentions_[i]->as<WanAttentionBlock>();
      if (attn) {
        x = attn->forward(x);
      }
      auto resnet = resnets_[i + 1]->as<WanResidualBlock>();
      x = resnet->forward(x, feat_cache, feat_idx);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < attentions_->size(); ++i) {
      attentions_[i]->as<WanAttentionBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("attentions." + std::to_string(i) +
                                          "."));
    }
  }

 private:
  int64_t dim_;
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList attentions_{nullptr};
};
TORCH_MODULE(WanMidBlock);

class WanResidualDownBlockImpl : public torch::nn::Module {
 public:
  WanResidualDownBlockImpl(int64_t in_dim,
                           int64_t out_dim,
                           float dropout,
                           int num_res_blocks,
                           bool temperal_downsample = false,
                           bool down_flag = false)
      : in_dim_(in_dim),
        out_dim_(out_dim),
        dropout_(dropout),
        num_res_blocks_(num_res_blocks),
        temperal_downsample_(temperal_downsample),
        down_flag_(down_flag) {
    int factor_t = temperal_downsample ? 2 : 1;
    int factor_s = down_flag ? 2 : 1;
    avg_shortcut_ = register_module(
        "avg_shortcut", AvgDown3D(in_dim, out_dim, factor_t, factor_s));
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int cur_in_dim = in_dim;
    for (int i = 0; i < num_res_blocks; ++i) {
      resnets_->push_back(WanResidualBlock(cur_in_dim, out_dim, dropout));
      cur_in_dim = out_dim;
    }
    if (down_flag) {
      std::string mode = temperal_downsample ? "downsample3d" : "downsample2d";
      downsampler_ =
          register_module("downsampler", WanResample(out_dim, mode, -1));
    } else {
      downsampler_ = nullptr;
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0}) {
    torch::Tensor x_copy = x.clone();
    for (size_t i = 0; i < resnets_->size(); ++i) {
      x = resnets_[i]->as<WanResidualBlock>()->forward(x, feat_cache, feat_idx);
    }
    if (downsampler_) {
      x = downsampler_->as<WanResample>()->forward(x, feat_cache, feat_idx);
    }
    return x + avg_shortcut_->forward(x_copy);
  }

  void load_state_dict(const StateDict& state_dict) {
    avg_shortcut_->as<AvgDown3D>()->load_state_dict(
        state_dict.get_dict_with_prefix("avg_shortcut."));
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (downsampler_) {
      downsampler_->as<WanResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("downsampler."));
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  float dropout_;
  int num_res_blocks_;
  bool temperal_downsample_, down_flag_;
  torch::nn::Module avg_shortcut_{nullptr};
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::Module downsampler_{nullptr};
};
TORCH_MODULE(WanResidualDownBlock);

class WanVAEEncoder3DImpl : public torch::nn::Module {
 public:
  WanVAEEncoder3DImpl(int64_t in_channels = 3,
                      int64_t dim = 128,
                      int64_t z_dim = 4,
                      std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                      int num_res_blocks = 2,
                      std::vector<float> attn_scales = {},
                      std::vector<bool> temperal_downsample = {true,
                                                               true,
                                                               false},
                      float dropout = 0.0f,
                      bool is_residual = false) {
    nonlinearity_ = torch::nn::Functional(torch::silu);
    std::vector<int64_t> dims;
    dims.push_back(dim);
    for (auto u : dim_mult) dims.push_back(dim * u);
    double scale = 1.0;
    conv_in_ = register_module("conv_in",
                               WanCausalConv3D(in_channels,
                                               dims[0],
                                               std::vector<int64_t>{3, 3, 3},
                                               std::vector<int64_t>{1, 1, 1},
                                               std::vector<int64_t>{1, 1, 1}));
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; ++i) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];
      if (is_residual) {
        down_blocks_->push_back(WanResidualDownBlock(
            in_dim,
            out_dim,
            dropout,
            num_res_blocks,
            (i != dim_mult.size() - 1) ? temperal_downsample[i] : false,
            (i != dim_mult.size() - 1)));
      } else {
        for (int j = 0; j < num_res_blocks; ++j) {
          int current_dim = in_dim;
          down_blocks_->push_back(
              WanResidualBlock(current_dim, out_dim, dropout));
          if (std::find(attn_scales.begin(), attn_scales.end(), scale) !=
              attn_scales.end()) {
            down_blocks_->push_back(WanAttentionBlock(out_dim));
          }
          current_dim = out_dim;
        }
        if (i != dim_mult.size() - 1) {
          std::string mode =
              temperal_downsample[i] ? "downsample3d" : "downsample2d";
          down_blocks_->push_back(WanResample(out_dim, mode, -1));
          scale /= 2.0;
        }
      }
    }
    mid_block_ =
        register_module("mid_block", WanMidBlock(dims.back(), dropout, 1));
    norm_out_ = register_module("norm_out",
                                WanRMSNorm(dims.back(), true, false, false));
    conv_out_ = register_module("conv_out",
                                WanCausalConv3D(dims.back(),
                                                z_dim,
                                                std::vector<int64_t>{3, 3, 3},
                                                std::vector<int64_t>{1, 1, 1},
                                                std::vector<int64_t>{1, 1, 1}));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0}) {
    if (feat_cache) {
      int idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv_in_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_in_->forward(x);
    }
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      if (feat_cache) {
        x = down_blocks_[i]->forward(x, feat_cache, feat_idx);
      } else {
        x = down_blocks_[i]->forward(x);
      }
    }
    x = mid_block_->forward(x, feat_cache, feat_idx);
    x = norm_out_->forward(x);
    x = nonlinearity_(x);
    if (feat_cache) {
      int idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv_out_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_out_->forward(x);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      down_blocks_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "down_blocks." + std::to_string(i) + "."));
    }
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

 private:
  torch::nn::Functional nonlinearity_{nullptr};
  torch::nn::Module conv_in_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  WanMidBlock mid_block_{nullptr};
  WanRMSNorm norm_out_{nullptr};
  WanCausalConv3D conv_out_{nullptr};
  const int CACHE_T = 2;
};
TORCH_MODULE(WanVAEEncoder3D);

class WanResidualUpBlockImpl : public torch::nn::Module {
 public:
  WanResidualUpBlockImpl(int64_t in_dim,
                         int64_t out_dim,
                         int num_res_blocks,
                         float dropout = 0.0f,
                         bool temperal_upsample = false,
                         bool up_flag = false)
      : in_dim_(in_dim), out_dim_(out_dim), num_res_blocks_(num_res_blocks) {
    if (up_flag_) {
      int factor_t = temperal_upsample ? 2 : 1;
      int factor_s = 2;
      avg_shortcut_ = register_module(
          "avg_shortcut", DupUp3D(in_dim, out_dim, factor_t, factor_s));
    } else {
      avg_shortcut_ = nullptr;
    }
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int current_dim = in_dim;
    for (int i = 0; i < num_res_blocks + 1; ++i) {
      resnets_->push_back(WanResidualBlock(current_dim, out_dim, dropout));
      current_dim = out_dim;
    }
    if (up_flag) {
      std::string upsample_mode =
          temperal_upsample ? "upsample3d" : "upsample2d";
      upsampler_ = register_module(
          "upsampler", WanResample(out_dim, upsample_mode, out_dim));
    } else {
      upsampler_ = nullptr;
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>({0}),
                        bool first_chunk = false) {
    torch::Tensor x_copy = x.clone();
    for (size_t i = 0; i < resnets_->size(); ++i) {
      if (feat_cache) {
        x = resnets_[i]->as<WanResidualBlock>()->forward(
            x, feat_cache, feat_idx);
      } else {
        x = resnets_[i]->as<WanResidualBlock>()->forward(x);
      }
    }
    if (upsampler_) {
      if (feat_cache) {
        x = upsampler_->as<WanResample>()->forward(x, feat_cache, feat_idx);
      } else {
        x = upsampler_->as<WanResample>()->forward(x);
      }
    }
    if (avg_shortcut_) {
      x = x + avg_shortcut_->as<DupUp3D>()->forward(x_copy, first_chunk);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (avg_shortcut_) {
      avg_shortcut_->as<DupUp3D>()->load_state_dict(
          state_dict.get_dict_with_prefix("avg_shortcut."));
    }
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (upsampler_) {
      upsampler_->as<WanResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("upsampler."));
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  int num_res_blocks_;
  torch::nn::Module avg_shortcut_{nullptr};
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::Module upsampler_{nullptr};
};
TORCH_MODULE(WanResidualUpBlock);

class WanUpBlockImpl : public torch::nn::Module {
 public:
  WanUpBlockImpl(int64_t in_dim,
                 int64_t out_dim,
                 int num_res_blocks,
                 float dropout = 0.0f,
                 const std::optional<std::string>& upsample_mode = std::nullopt)
      : in_dim_(in_dim), out_dim_(out_dim), num_res_blocks_(num_res_blocks) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int current_dim = in_dim;
    for (int i = 0; i < num_res_blocks + 1; ++i) {
      resnets_->push_back(WanResidualBlock(current_dim, out_dim, dropout));
      current_dim = out_dim;
    }
    if (upsample_mode.has_value()) {
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsamplers_->push_back(WanResample(out_dim, upsample_mode.value()));
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0},
                        torch::Tensor* first_chunk = nullptr) {
    torch::Tensor h = x;
    for (size_t i = 0; i < resnets_->size(); ++i) {
      auto resnet = resnets_[i]->as<WanResidualBlock>();
      if (feat_cache) {
        h = resnet->forward(h, *feat_cache, *feat_idx, first_chunk);
      } else {
        h = resnet->forward(h);
      }
    }
    if (upsamplers_ && upsamplers_->size() > 0) {
      auto upsampler = upsamplers_[0]->as<WanResample>();
      if (feat_cache) {
        h = upsampler->forward(h, *feat_cache, *feat_idx, first_chunk);
      } else {
        h = upsampler->forward(h);
      }
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (upsamplers_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        upsamplers_[i]->as<WanResample>()->load_state_dict(
            state_dict.get_dict_with_prefix("upsamplers." + std::to_string(i) +
                                            "."));
      }
    }
  }

 private:
  int64_t in_dim_;
  int64_t out_dim_;
  int num_res_blocks_;
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
};
TORCH_MODULE(WanUpBlock);

class WanVAEDecoder3DImpl : public torch::nn::Module {
 public:
  WanVAEDecoder3DImpl(int64_t dim = 128,
                      int64_t z_dim = 4,
                      const std::vector<int64_t>& dim_mult = {1, 2, 4, 4},
                      int num_res_blocks = 2,
                      const std::vector<float>& attn_scales = {},
                      const std::vector<bool>& temperal_upsample = {false,
                                                                    true,
                                                                    true},
                      float dropout = 0.0f,
                      int64_t out_channels = 3,
                      bool is_residual = false) {
    std::vector<int64_t> dims;
    dims.push_back(dim * dim_mult.back());
    for (auto it = dim_mult.rbegin(); it != dim_mult.rend(); ++it) {
      dims.push_back(dim * (*it));
    }
    dims.erase(dims.begin());
    conv_in_ = register_module("conv_in",
                               WanCausalConv3d(z_dim,
                                               dims[0],
                                               std::vector<int64_t>{3, 3, 3},
                                               std::vector<int64_t>{1, 1, 1},
                                               std::vector<int64_t>{1, 1, 1}));
    mid_block_ =
        register_module("mid_block", WanMidBlock(dims[0], dropout_, 1));
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; ++i) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];
      if (i > 0 && !is_residual) {
        in_dim = in_dim / 2;
      }
      bool up_flag = (i != dim_mult.size() - 1);
      std::string upsample_mode;
      if (up_flag && temperal_upsample[i]) {
        upsample_mode = "upsample3d";
      } else if (up_flag) {
        upsample_mode = "upsample2d";
      }
      if (is_residual) {
        up_blocks_->push_back(
            WanResidualUpBlock(in_dim,
                               out_dim,
                               num_res_blocks,
                               dropout,
                               (up_flag ? temperal_upsample[i] : false),
                               up_flag));
      } else {
        up_blocks_->push_back(
            WanUpBlock(in_dim,
                       out_dim,
                       num_res_blocks,
                       dropout,
                       up_flag ? std::optional<std::string>(upsample_mode)
                               : std::nullopt));
      }
    }
    nonlinearity_ = torch::nn::Functional(torch::silu);
    norm_out_ = register_module("norm_out",
                                WanRMSNorm(dims.back(), true, false, false));
    conv_out_ = register_module("conv_out",
                                WanCausalConv3d(dims.back(),
                                                out_channels_,
                                                std::vector<int64_t>{3, 3, 3},
                                                std::vector<int64_t>{1, 1, 1},
                                                std::vector<int64_t>{1, 1, 1}));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int>* feat_idx = std::vector<int>{0},
                        bool first_chunk = false) {
    // conv_in
    if (feat_cache) {
      int idx = (*feat_idx)[0];
      torch::Tensor cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].defined()) {
        cache_x = torch::cat(
            {(*feat_cache)[idx]
                 .index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(-1, torch::indexing::None),
                         torch::indexing::Slice(),
                         torch::indexing::Slice()})
                 .unsqueeze(2)
                 .to(cache_x.device()),
             cache_x},
            2);
      }
      x = conv_in_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_in_->forward(x);
    }

    // mid_block
    x = mid_block_->forward(x, feat_cache, feat_idx);

    // up_blocks
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      auto up_block = up_blocks_[i];
      x = up_block->as<WanUpBlock>()->forward(
          x, feat_cache, feat_idx, &first_chunk);
    }

    x = norm_out_->forward(x);
    x = nonlinearity_(x);

    // conv_out
    if (feat_cache) {
      int idx = (*feat_idx)[0];
      torch::Tensor cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].defined()) {
        cache_x = torch::cat(
            {(*feat_cache)[idx]
                 .index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(-1, torch::indexing::None),
                         torch::indexing::Slice(),
                         torch::indexing::Slice()})
                 .unsqueeze(2)
                 .to(cache_x.device()),
             cache_x},
            2);
      }
      x = conv_out_->forward(x, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_out_->forward(x);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      up_blocks_[i]->as<WanUpBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("up_blocks." + std::to_string(i) +
                                          "."));
    }
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

 private:
  WanCausalConv3d conv_in_{nullptr};
  WanMidBlock mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_{nullptr};
  WanRMSNorm norm_out_{nullptr};
  WanCausalConv3d conv_out_{nullptr};
  torch::nn::Functional nonlinearity_{nullptr};
  const int CACHE_T = 2;
};
TORCH_MODULE(WanVAEDecoder3D);

class WANVAEImpl : public torch::nn::Module {
 public:
  WANVAEImpl(const ModelContext& context,
             torch::Device device,
             torch::ScalarType dtype)
      : args_(context.get_model_args()), device_(device), dtype_(dtype) {
    encoder_ = register_module("encoder",
                               WanVAEEncoder3D(args_.vae_in_channels(),
                                               args_.vae_base_dim(),
                                               args_.vae_z_dim() * 2,
                                               args_.vae_dim_mult(),
                                               args_.vae_num_res_blocks(),
                                               args_.vae_attn_scales(),
                                               args_.vae_temperal_downsample(),
                                               args_.vae_dropout(),
                                               args_.vae_is_residual()));

    decoder_ = register_module("decoder",
                               WanVAEDecoder3D(args_.vae_base_dim(),
                                               args_.vae_z_dim(),
                                               args_.vae_dim_mult(),
                                               args_.vae_num_res_blocks(),
                                               args_.vae_attn_scales(),
                                               args_.vae_temperal_downsample(),
                                               args_.vae_dropout(),
                                               args_.vae_is_residual()));

    quant_conv_ =
        register_module("quant_conv",
                        WanCausalConv3D(2 * args_.z_dim(),
                                        2 * args_.z_dim(),
                                        std::vector<int64_t>{1, 1, 1}));

    post_quant_conv_ = register_module(
        "post_quant_conv",
        WanCausalConv3D(
            args_.z_dim(), args_.z_dim(), std::vector<int64_t>{1, 1, 1}));
    init_cached_conv_count();

    encoder_->to(dtype_);
    decoder_->to(dtype_);
    quant_conv_->to(dtype_);
    post_quant_conv_->to(dtype_);
  }

  void enable_slicing(bool enable) { use_slicing_ = enable; }
  void disable_slicing() { use_slicing_ = false; }

  void clear_cache() {
    conv_num_ = cached_conv_count_["decoder"];
    conv_idx_ = {0};
    feat_map_.assign(conv_num_, nullptr);

    enc_conv_num_ = cached_conv_count_["encoder"];
    enc_conv_idx_ = {0};
    enc_feat_map_.assign(enc_conv_num_, nullptr);
  }

  // Encode video into latent representations
  torch::Tensor encode_(const torch::Tensor& videos) {
    int64_t num_frame = videos.size(2);
    int64_t height = videos.size(3);
    int64_t width = videos.size(4);
    int64_t iter_ = 1 + (num_frame - 1) / 4;
    clear_cache();
    torch::Tensor out;
    for (int64_t i = 0; i < iter_; ++i) {
      enc_conv_idx_ = {0};
      if (i == 0) {
        auto x_slice = videos.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(0, 1),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
        out = encoder_(x_slice, &enc_feat_map_, &enc_conv_idx_);
      } else {
        int64_t start = 1 + 4 * (i - 1);
        int64_t end = std::min(1 + 4 * i, num_frame);
        auto x_slice = videos.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(start, end),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
        auto out_ = encoder_(x_slice, &enc_feat_map_, &enc_conv_idx_);
        out = torch::cat({out, out_}, 2);
      }
    }
    out = quant_conv_(out);
    clear_cache();
    return out;
  }

  AutoencoderKLOutput encode(const torch::Tensor& videos) {
    torch::Tensor hidden_states;
    if (use_slicing_) {
      std::vector<torch::Tensor> latent_slices;
      for (const auto& x_slice : videos.split(1)) {
        latent_slices.push_back(encode_(x_slice));
      }
      hidden_states = torch::cat(latent_slices, 0);
    } else {
      hidden_states = encode_(videos);
    }
    auto posterior = DiagonalGaussianDistribution(hidden_states);
    return AutoencoderKLOutput(posterior);
  }

  // Decode latent representations into videos
  DecoderOutput decode_(const torch::Tensor& latents) {
    torch::Tensor processed_latents = latents;
    int64_t num_frame = latents.size(2);
    int64_t height = latents.size(3);
    int64_t width = latents.size(4);
    clear_cache();
    torch::Tensor out;
    processed_latents = post_quant_conv_(processed_latents);
    for (int64_t i = 0; i < iter_; ++i) {
      enc_conv_idx_ = {0};
      if (i == 0) {
        auto x_slice =
            processed_latents.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(i, i + 1),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
        out = decoder_(x_slice, &enc_feat_map_, &enc_conv_idx_, true);
      } else {
        auto x_slice =
            processed_latents.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(i, i + 1),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
        auto out_ = decoder_(x_slice, &enc_feat_map_, &enc_conv_idx_);
        out = torch::cat({out, out_}, 2);
      }
    }
    auto dec = torch::clamp(out, -1.0f, 1.0f);
    clear_cache();
    return DecoderOutput(dec);
  }

  DecoderOutput decode(
      const torch::Tensor& latents,
      const std::optional<torch::Generator>& generator = std::nullopt) {
    torch::Tensor videos;
    if (use_slicing_ && latents.size(0) > 1) {
      std::vector<torch::Tensor> video_slices;
      for (const auto& latent_slice : latents.split(1)) {
        video_slices.push_back(decode_(latent_slice).sample);
      }
      videos = torch::cat(video_slices, 0);
    } else {
      videos = decode_(latents).sample;
    }
    return DecoderOutput(videos);
  }

  DecoderOutput forward_(torch::Tensor sample, bool sample_posterior = false) {
    torch::Tensor x = sample;
    DiagonalGaussianDistribution posterior = encode(x).latent_dist;

    if (sample_posterior) {
      x = posterior.sample();
    } else {
      x = posterior.mode();
    }

    return decode(x);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));

      if (args_.vae_use_quant_conv()) {
        const auto weight = state_dict->get_tensor("quant_conv.weight");
        if (weight.defined()) {
          DCHECK_EQ(quant_conv_->weight.sizes(), weight.sizes())
              << "quant_conv weight size mismatch";
          quant_conv_->weight.data().copy_(weight);
          is_quant_conv_loaded = true;
        }

        const auto bias = state_dict->get_tensor("quant_conv.bias");
        if (bias.defined() && quant_conv_->bias.defined()) {
          DCHECK_EQ(quant_conv_->bias.sizes(), bias.sizes())
              << "quant_conv bias size mismatch";
          quant_conv_->bias.data().copy_(bias);
        }
      }

      if (args_.vae_use_post_quant_conv()) {
        const auto weight = state_dict->get_tensor("post_quant_conv.weight");
        if (weight.defined()) {
          DCHECK_EQ(post_quant_conv_->weight.sizes(), weight.sizes())
              << "post_quant_conv weight size mismatch";
          post_quant_conv_->weight.data().copy_(weight);
          is_post_quant_conv_loaded = true;
        }

        const auto bias = state_dict->get_tensor("post_quant_conv.bias");
        if (bias.defined() && post_quant_conv_->bias.defined()) {
          DCHECK_EQ(post_quant_conv_->bias.sizes(), bias.sizes())
              << "post_quant_conv bias size mismatch";
          post_quant_conv_->bias.data().copy_(bias);
        }
      }
    }
    LOG(INFO) << "WAN VAE model loaded successfully.";
  }

 private:
  WanVAEEncoder3D encoder_{nullptr};
  WanVAEDecoder3D decoder_{nullptr};
  torch::nn::Conv3d quant_conv_{nullptr};
  torch::nn::Conv3d post_quant_conv_{nullptr};
  bool use_slicing_{false};
  ModelArgs args_;
  torch::Device device_;
  torch::ScalarType dtype_;
  int64_t tile_sample_min_height_ = 256;
  int64_t tile_sample_min_width_ = 256;
  int64_t tile_sample_stride_height_ = 192;
  int64_t tile_sample_stride_width_ = 192;
  std::map<std::string, int> cached_conv_count_;
  int conv_num_ = 0;
  std::vector<int> conv_idx_{0};
  std::vector<torch::Tensor> feat_map_;
  int enc_conv_num_ = 0;
  std::vector<int> enc_conv_idx_{0};
  std::vector<torch::Tensor> enc_feat_map_;

  void init_cached_conv_count() {
    int decoder_count = 0;
    int encoder_count = 0;
    if (decoder_ != nullptr) {
      for (const auto& m : decoder_->modules(/*include_self=*/false)) {
        if (dynamic_cast<WanCausalConv3DImpl*>(m.get()) != nullptr) {
          ++decoder_count;
        }
      }
    }
    if (encoder_ != nullptr) {
      for (const auto& m : encoder_->modules(/*include_self=*/false)) {
        if (dynamic_cast<WanCausalConv3DImpl*>(m.get()) != nullptr) {
          ++encoder_count;
        }
      }
    }
    cached_conv_count_["decoder"] = decoder_count;
    cached_conv_count_["encoder"] = encoder_count;
  }
};
TORCH_MODULE(WANVAE);

REGISTER_MODEL_ARGS(WANVAE, [&] {
  LOAD_ARG_OR(vae_z_dim, "z_dim", 16);
  LOAD_ARG_OR(vae_base_dim, "base_dim", 96);
  LOAD_ARG_OR(vae_num_res_blocks, "num_res_blocks", 2);
  LOAD_ARG_OR(vae_temperal_downsample,
              "temporal_downsample",
              std::vector<bool>{false, true, true});
  LOAD_ARG_OR(vae_attn_scale, "attn_scale", std::vector<float>{});
  LOAD_ARG_OR(vae_dim_mults, "dim_mults", std::vector<int>{1, 2, 4, 4});
  LOAD_ARG_OR(vae_dropout, "dropout", 0.0f);
  LOAD_ARG_OR(vae_in_channels, "in_channels", 3);
  LOAD_ARG_OR(vae_out_channels, "out_channels", 3);
  LOAD_ARG_OR(vae_is_residual, "is_residual", false);
  LOAD_ARG_OR(vae_scale_factor_temporal, "scale_factor_temporal", 4);
  LOAD_ARG_OR(vae_scale_factor_spatial, "scale_factor_spatial", 8);
  LOAD_ARG_OR(vae_latents_mean,
              "latents_mean",
              std::vector<float>{-0.7571,
                                 -0.7089,
                                 -0.9113,
                                 0.1075,
                                 -0.1745,
                                 0.9653,
                                 -0.1517,
                                 1.5508,
                                 0.4134,
                                 -0.0715,
                                 0.5517,
                                 -0.3632,
                                 -0.1922,
                                 -0.9497,
                                 0.2503,
                                 -0.2921});
  LOAD_ARG_OR(vae_latents_std,
              "latents_std",
              std::vector<float>{2.8184,
                                 1.4541,
                                 2.3275,
                                 2.6558,
                                 1.2196,
                                 1.7708,
                                 2.6052,
                                 2.0743,
                                 3.2687,
                                 2.1526,
                                 2.8652,
                                 1.5579,
                                 1.6382,
                                 1.1253,
                                 2.8251,
                                 1.916});
});

}  // namespace xllm
