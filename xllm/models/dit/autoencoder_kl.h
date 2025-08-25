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

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
// VAE model compatible with huggingface weights
//  ref to:
//  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py

namespace xllm {
class VAEImageProcessorImpl : public torch::nn::Module {
 private:
  int vae_scale_factor_ = 8;
  int vae_latent_channels_ = 4;
  bool do_resize_ = true;
  bool do_normalize_ = true;
  bool do_binarize_ = false;
  bool do_convert_rgb_ = false;
  bool do_convert_grayscale_ = false;
  std::string resample_ = "lanczos";
  int reducing_gap_ = -1;  // not used
 public:
  VAEImageProcessorImpl(bool do_resize = true,
                        int vae_scale_factor = 8,
                        int vae_latent_channels = 4,
                        std::string resample = "lanczos",
                        int reducing_gap = -1,
                        bool do_normalize = true,
                        bool do_binarize = false,
                        bool do_convert_rgb = false,
                        bool do_convert_grayscale = false)
      : do_resize_(do_resize),
        vae_scale_factor_(vae_scale_factor),
        vae_latent_channels_(vae_latent_channels),
        resample_(resample),
        reducing_gap_(reducing_gap),
        do_normalize_(do_normalize),
        do_binarize_(do_binarize),
        do_convert_rgb_(do_convert_rgb),
        do_convert_grayscale_(do_convert_grayscale) {}
  std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
                                                int64_t width) const {
    height = height - (height % vae_scale_factor_);
    width = width - (width % vae_scale_factor_);
    return {height, width};
  }
  torch::Tensor normalize(const torch::Tensor& tensor) const {
    return 2.0 * tensor - 1.0;
  }
  torch::Tensor denormalize(const torch::Tensor& tensor) const {
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0);
  }
  torch::Tensor resize(const torch::Tensor& image,
                       int64_t target_height,
                       int64_t target_width) const {
    return torch::nn::functional::interpolate(
        image,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height, target_width})
            .align_corners(false));
  }
  std::pair<int64_t, int64_t> get_default_height_width(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt) const {
    int64_t h, w;
    if (image.dim() == 3) {
      h = image.size(1);
      w = image.size(2);
    } else if (image.dim() == 4) {
      h = image.size(2);
      w = image.size(3);
    } else {
      throw std::invalid_argument("Invalid image tensor dimensions");
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);
    return adjust_dimensions(target_h, target_w);
  }
  torch::Tensor preprocess(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      const std::string& resize_mode = "default",
      std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          crop_coords = std::nullopt) {
    torch::Tensor processed = image.clone();
    if (processed.dtype() != torch::kFloat32) {
      processed = processed.to(torch::kFloat32);
    }
    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }
    if (crop_coords.has_value()) {
      auto [x1, y1, x2, y2] = crop_coords.value();
      x1 = std::max(int64_t(0), x1);
      y1 = std::max(int64_t(0), y1);
      x2 = std::min(processed.size(-1), x2);
      y2 = std::min(processed.size(-2), y2);

      if (processed.dim() == 3) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      } else if (processed.dim() == 4) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      }
    }
    if (do_convert_grayscale_ && processed.size(1) == 3) {
      std::vector<float> weights = {0.299f, 0.587f, 0.114f};
      torch::Tensor weight_tensor =
          torch::tensor(weights, torch::kFloat32).view({1, 3, 1, 1});
      if (processed.dim() == 3) {
        weight_tensor = weight_tensor.squeeze(0);
      }
      processed = torch::sum(processed * weight_tensor, 1, true);
    } else if (do_convert_rgb_ && processed.size(1) == 1) {
      processed = torch::cat({processed, processed, processed}, 1);
    }

    auto [target_h, target_w] =
        get_default_height_width(processed, height, width);
    if (do_resize_) {
      processed = resize(processed, target_h, target_w);
    }

    if (do_normalize_) {
      processed = normalize(processed);
    }
    if (do_binarize_) {
      processed = (processed >= 0.5f).to(torch::kFloat32);
    }

    return processed;
  }
  torch::Tensor postprocess(
      const torch::Tensor& tensor,
      const std::string& output_type = "pt",
      std::optional<std::vector<bool>> do_denormalize = std::nullopt) {
    torch::Tensor processed = tensor.clone();
    if (do_normalize_) {
      if (!do_denormalize.has_value()) {
        processed = denormalize(processed);
      } else {
        for (int64_t i = 0; i < processed.size(0); ++i) {
          if (i < do_denormalize.value().size() && do_denormalize.value()[i]) {
            processed[i] = denormalize(processed[i]);
          }
        }
      }
    }
    if (output_type == "np") {
      return processed.permute({0, 2, 3, 1}).contiguous();
    }
    return processed;
  }
};
TORCH_MODULE(VAEImageProcessor);
class SpatialNormImpl : public torch::nn::Module {
 public:
  SpatialNormImpl(int64_t f_channels, int64_t zq_channels) {
    norm_layer = register_module(
        "norm_layer",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(f_channels, 32).eps(1e-6)));

    conv_y = register_module("conv_y",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                 zq_channels, f_channels, 1)));

    conv_b = register_module("conv_b",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                 zq_channels, f_channels, 1)));
  }

  torch::Tensor forward(torch::Tensor f, torch::Tensor zq) {
    auto f_size = std::vector<int64_t>{f.size(2), f.size(3)};
    zq = torch::nn::functional::interpolate(
        zq,
        torch::nn::functional::InterpolateFuncOptions()
            .size(f_size)
            .mode(torch::kBilinear)
            .align_corners(false));

    torch::Tensor norm_f = norm_layer(f);
    torch::Tensor new_f = norm_f * conv_y(zq) + conv_b(zq);
    return new_f;
  }
  void load_state_dict(const StateDict& state_dict) {
    // norm_layer
    const auto norm_layer_state_weight =
        state_dict.get_tensor("norm_layer.weight");
    if (norm_layer_state_weight.defined()) {
      norm_layer->weight.data().copy_(norm_layer_state_weight);
    }
    const auto norm_layer_state_bias = state_dict.get_tensor("norm_layer.bias");
    if (norm_layer_state_bias.defined()) {
      norm_layer->bias.data().copy_(norm_layer_state_bias);
    }
    // conv_y
    const auto conv_y_state_bias = state_dict.get_tensor("conv_y.bias");
    if (conv_y_state_bias.defined()) {
      conv_y->bias.data().copy_(conv_y_state_bias);
    }
    const auto conv_y_state_weight = state_dict.get_tensor("conv_y.weight");
    if (conv_y_state_weight.defined()) {
      conv_y->weight.data().copy_(conv_y_state_weight);
    }
    // conv_b
    const auto conv_b_state_bias = state_dict.get_tensor("conv_b.bias");
    if (conv_b_state_bias.defined()) {
      conv_b->bias.data().copy_(conv_b_state_bias);
    }
    const auto conv_b_state_weight = state_dict.get_tensor("conv_b.weight");
    if (conv_b_state_weight.defined()) {
      conv_b->weight.data().copy_(conv_b_state_weight);
    }
  }

 private:
  torch::nn::GroupNorm norm_layer{nullptr};
  torch::nn::Conv2d conv_y{nullptr};
  torch::nn::Conv2d conv_b{nullptr};
};
TORCH_MODULE(SpatialNorm);

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int64_t query_dim,
                int64_t num_heads,
                int64_t head_dim,
                float rescale_output_factor = 1.0f,
                float eps = 1e-6f,
                int64_t norm_num_groups = 32,
                std::optional<int64_t> spatial_norm_dim = std::nullopt,
                bool residual_connection = true,
                bool bias = true,
                bool upcast_softmax = true,
                bool _from_deprecated_attn_block = false)
      : num_heads_(num_heads),
        rescale_output_factor_(rescale_output_factor),
        residual_connection_(residual_connection) {
    int64_t spatial_norm_dim_val;
    if (spatial_norm_dim.has_value()) {
      spatial_norm_dim_val = spatial_norm_dim.value();
    } else {
      spatial_norm_dim_val = -1;
    }
    if (spatial_norm_dim_val > 0) {
      spatial_norm_ = register_module(
          "spatial_norm", SpatialNorm(query_dim, spatial_norm_dim_val));
    }
    if (norm_num_groups > 0) {
      group_norm_ =
          register_module("group_norm",
                          torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                   norm_num_groups, query_dim)
                                                   .eps(eps)));
    }
    int64_t inner_dim = head_dim * num_heads;
    to_q_ = register_module("to_q", DiTLinear(query_dim, inner_dim, true));
    to_k_ = register_module("to_k", DiTLinear(query_dim, inner_dim, true));
    to_v_ = register_module("to_v", DiTLinear(query_dim, inner_dim, true));
    to_out_ = register_module("to_out", DiTLinear(inner_dim, query_dim, true));
  }
  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor temb) {
    torch::Tensor residual = hidden_states;
    int64_t input_ndim = hidden_states.dim();
    int64_t batch_size;
    int64_t channel;
    int64_t height;
    int64_t width;
    if (input_ndim == 4) {
      batch_size = hidden_states.size(0);
      channel = hidden_states.size(1);
      height = hidden_states.size(2);
      width = hidden_states.size(3);

      // reshape tensor：[B, C, H, W] -> [B, C, H*W]
      hidden_states = hidden_states.view({batch_size, channel, height * width});

      // swap dim 1&2：[B, C, H*W] -> [B, H*W, C]
      hidden_states = hidden_states.transpose(1, 2);
    }
    if (temb.defined()) {
      torch::IntArrayRef input_shape = temb.sizes();
      batch_size = input_shape[0];
      int64_t seq_length = input_shape[1];
    }
    hidden_states =
        group_norm_->forward(hidden_states.transpose(1, 2)).transpose(1, 2);
    torch::Tensor query = to_q_(hidden_states);
    // if temb is none
    if (!temb.defined()) {
      temb = hidden_states.clone();
    }
    torch::Tensor key = to_k_(temb);
    torch::Tensor value = to_v_(temb);
    int64_t inner_dim = key.size(-1);
    int64_t head_dim = inner_dim / num_heads_;
    query = query.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    hidden_states = at::scaled_dot_product_attention(query,
                                                     key,
                                                     value,
                                                     c10::nullopt,  // attn_mask
                                                     0.0,           // dropout_p
                                                     false          // is_causal
    );
    hidden_states = hidden_states.transpose(1, 2).reshape(
        {batch_size, -1, num_heads_ * head_dim});
    hidden_states = hidden_states.to(query.dtype());
    hidden_states = to_out_(hidden_states);
    // alignment right
    if (input_ndim == 4) {
      hidden_states = hidden_states.transpose(-1, -2).reshape(
          {batch_size, channel, height, width});
    }
    if (residual_connection_ == true) {
      hidden_states = hidden_states + residual;
    }
    hidden_states = hidden_states / rescale_output_factor_;
    return hidden_states;
  }
  void load_state_dict(const StateDict& state_dict) {
    // to_q_
    const auto to_q_state_bias = state_dict.get_tensor("to_q.bias");
    if (to_q_state_bias.defined()) {
      DCHECK_EQ(to_q_->bias.sizes(), to_q_state_bias.sizes())
          << "to_q bias size mismatch: expected " << to_q_->bias.sizes()
          << " but got " << to_q_state_bias.sizes();
      to_q_->bias.data().copy_(to_q_state_bias);
    }
    const auto to_q_state_weight = state_dict.get_tensor("to_q.weight");
    if (to_q_state_weight.defined()) {
      DCHECK_EQ(to_q_->weight.sizes(), to_q_state_weight.sizes())
          << "to_q weight size mismatch: expected " << to_q_->weight.sizes()
          << " but got " << to_q_state_weight.sizes();
      to_q_->weight.data().copy_(to_q_state_weight);
    }
    // to_k_
    const auto to_k_state_bias = state_dict.get_tensor("to_k.bias");
    if (to_k_state_bias.defined()) {
      DCHECK_EQ(to_k_->bias.sizes(), to_k_state_bias.sizes())
          << "to_k bias size mismatch: expected " << to_k_->bias.sizes()
          << " but got " << to_k_state_bias.sizes();
      to_k_->bias.data().copy_(to_k_state_bias);
    }
    const auto to_k_state_weight = state_dict.get_tensor("to_k.weight");
    if (to_k_state_weight.defined()) {
      DCHECK_EQ(to_k_->weight.sizes(), to_k_state_weight.sizes())
          << "to_k weight size mismatch: expected " << to_k_->weight.sizes()
          << " but got " << to_k_state_weight.sizes();
      to_k_->weight.data().copy_(to_k_state_weight);
    }
    // to_v_
    const auto to_v_state_bias = state_dict.get_tensor("to_v.bias");
    if (to_v_state_bias.defined()) {
      DCHECK_EQ(to_v_->bias.sizes(), to_v_state_bias.sizes())
          << "to_v bias size mismatch: expected " << to_v_->bias.sizes()
          << " but got " << to_v_state_bias.sizes();
      to_v_->bias.data().copy_(to_v_state_bias);
    }
    const auto to_v_state_weight = state_dict.get_tensor("to_v.weight");
    if (to_v_state_weight.defined()) {
      DCHECK_EQ(to_v_->weight.sizes(), to_v_state_weight.sizes())
          << "to_v weight size mismatch: expected " << to_v_->weight.sizes()
          << " but got " << to_v_state_weight.sizes();
      to_v_->weight.data().copy_(to_v_state_weight);
    }
    // to_out_
    const auto to_out_state_bias = state_dict.get_tensor("to_out.0.bias");
    if (to_out_state_bias.defined()) {
      DCHECK_EQ(to_out_->bias.sizes(), to_out_state_bias.sizes())
          << "to_out bias size mismatch: expected " << to_out_->bias.sizes()
          << " but got " << to_out_state_bias.sizes();
      to_out_->bias.data().copy_(to_out_state_bias);
    }
    const auto to_out_state_weight = state_dict.get_tensor("to_out.0.weight");
    if (to_out_state_weight.defined()) {
      DCHECK_EQ(to_out_->weight.sizes(), to_out_state_weight.sizes())
          << "to_out weight size mismatch: expected " << to_out_->weight.sizes()
          << " but got " << to_out_state_weight.sizes();
      to_out_->weight.data().copy_(to_out_state_weight);
    }
    if (spatial_norm_) {
      spatial_norm_->load_state_dict(
          state_dict.get_dict_with_prefix("spatial_norm."));
    }
    if (group_norm_) {
      // group_norm_
      const auto group_norm_state_dict =
          state_dict.get_tensor("group_norm.weight");
      if (group_norm_state_dict.defined()) {
        DCHECK_EQ(group_norm_->weight.sizes(), group_norm_state_dict.sizes())
            << "group_norm weight size mismatch: expected "
            << group_norm_->weight.sizes() << " but got "
            << group_norm_state_dict.sizes();
        group_norm_->weight.data().copy_(group_norm_state_dict);
      }
      const auto group_norm_state_bias =
          state_dict.get_tensor("group_norm.bias");
      if (group_norm_state_bias.defined()) {
        DCHECK_EQ(group_norm_->bias.sizes(), group_norm_state_bias.sizes())
            << "group_norm bias size mismatch: expected "
            << group_norm_->bias.sizes() << " but got "
            << group_norm_state_bias.sizes();
        group_norm_->bias.data().copy_(group_norm_state_bias);
      }
    }
  }

 private:
  SpatialNorm spatial_norm_{nullptr};
  torch::nn::GroupNorm group_norm_{nullptr};
  DiTLinear to_q_{nullptr};
  DiTLinear to_k_{nullptr};
  DiTLinear to_v_{nullptr};
  DiTLinear to_out_{nullptr};
  int64_t num_heads_;
  bool residual_connection_;
  float rescale_output_factor_;
};
TORCH_MODULE(Attention);

class Downsample2DImpl : public torch::nn::Module {
 public:
  Downsample2DImpl(int64_t channels,
                   bool use_conv = false,
                   std::optional<int64_t> out_channels = std::nullopt,
                   int64_t padding = 1,
                   const std::string& name = "conv",
                   int64_t kernel_size = 3,
                   const std::optional<std::string>& norm_type = std::nullopt,
                   std::optional<float> eps = std::nullopt,
                   std::optional<bool> elementwise_affine = std::nullopt,
                   bool bias = true)
      : channels_(channels),
        out_channels_(out_channels.value_or(channels)),
        use_conv_(use_conv),
        padding_(padding),
        stride_(2),
        name_(name),
        kernel_size_(kernel_size) {
    float eps_val = eps.has_value() ? eps.value() : 1e-5f;
    bool affine_val =
        elementwise_affine.has_value() ? elementwise_affine.value() : true;

    // if (norm_type.has_value()) {
    //     const std::string& norm = norm_type.value();
    //     if (norm == "ln_norm") {
    //         norm_ = register_module(
    //             "norm",
    //             LayerNorm(
    //                 channels_,
    //                 eps_val,
    //                 affine_val
    //             )
    //         );
    //     } else if (norm == "rms_norm") {
    //         norm_ = register_module(
    //             "norm",
    //             RMSNorm(
    //                 channels_,
    //                 eps_val
    //             )
    //         );
    //     } else {
    //         throw std::invalid_argument("Unknown norm_type: " + norm);
    //     }
    // }

    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels_, out_channels_, kernel_size_)
                .stride(stride_)
                .padding(padding_)
                .bias(bias)));

    if (name == "conv") {
      register_module("Conv2d_0", conv_);
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<torch::Tensor>& args = {}) {
    // check input channels
    TORCH_CHECK(hidden_states.size(1) == channels_,
                "Input channels mismatch: expected ",
                channels_,
                " but got ",
                hidden_states.size(1));

    torch::Tensor x = hidden_states;
    // if (norm_) {
    //     // according to Python's permute(0,2,3,1) -> norm -> permute(0,3,1,2)
    //     x = x.permute({0, 2, 3, 1});  // (B, C, H, W) -> (B, H, W, C)
    //     x = norm_.forward(x);
    //     x = x.permute({0, 3, 1, 2});  // (B, H, W, C) -> (B, C, H, W)
    // }

    if (use_conv_ && padding_ == 0) {
      x = torch::nn::functional::pad(
          x,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1})
              .mode(torch::kConstant)
              .value(0.0f));
    }
    TORCH_CHECK(x.size(1) == channels_,
                "Channels mismatch after norm/pad: expected ",
                channels_,
                " but got ",
                x.size(1));

    x = conv_(x);
    return x;
  }
  void load_state_dict(const StateDict& state_dict) {
    const auto conv_state_dict = state_dict.get_tensor("conv.weight");
    if (conv_state_dict.defined()) {
      DCHECK_EQ(conv_->weight.sizes(), conv_state_dict.sizes())
          << "conv weight size mismatch: expected " << conv_->weight.sizes()
          << " but got " << conv_state_dict.sizes();
      conv_->weight.data().copy_(conv_state_dict);
    }
    const auto conv_state_bias = state_dict.get_tensor("conv.bias");
    if (conv_state_bias.defined()) {
      DCHECK_EQ(conv_->bias.sizes(), conv_state_bias.sizes())
          << "conv bias size mismatch: expected " << conv_->bias.sizes()
          << " but got " << conv_state_bias.sizes();
      conv_->bias.data().copy_(conv_state_bias);
    }
  }

 private:
  int64_t channels_;
  int64_t out_channels_;
  bool use_conv_;
  int64_t padding_;
  int64_t stride_;
  std::string name_;
  int64_t kernel_size_;

  // torch::nn::AnyModule norm_;
  torch::nn::Conv2d conv_{nullptr};
};

TORCH_MODULE(Downsample2D);
class Upsample2DImpl : public torch::nn::Module {
 public:
  Upsample2DImpl(int64_t channels,
                 bool use_conv = false,
                 bool use_conv_transpose = false,
                 std::optional<int64_t> out_channels = std::nullopt,
                 std::string name = "conv",
                 int64_t kernel_size = 3,
                 int64_t padding = 1,
                 const std::optional<std::string>& norm_type = std::nullopt,
                 double eps = 1e-5,
                 bool elementwise_affine = true,
                 bool bias = true,
                 bool interpolate = true)
      : channels_(channels),
        use_conv_(use_conv),
        use_conv_transpose_(use_conv_transpose),
        name_(std::move(name)),
        interpolate_(interpolate) {
    out_channels_ = out_channels.value_or(channels);
    if (use_conv_transpose_) {
      int64_t k_size = (kernel_size == -1) ? 4 : kernel_size;
      conv_transpose_ = register_module(
          "conv_transpose",
          torch::nn::ConvTranspose2d(
              torch::nn::ConvTranspose2dOptions(channels, out_channels_, k_size)
                  .stride(2)
                  .padding(padding)
                  .bias(bias)));
    }
    if (use_conv_) {
      int64_t k_size = (kernel_size == -1) ? 3 : kernel_size;
      conv_ =
          register_module("conv",
                          torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                                channels, out_channels_, k_size)
                                                .padding(padding)
                                                .bias(bias)));
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<int64_t>& output_size = {}) {
    TORCH_CHECK(hidden_states.size(1) == channels_,
                "输入通道不匹配: 预期",
                channels_,
                "，实际",
                hidden_states.size(1));

    torch::Tensor x = hidden_states;
    if (use_conv_transpose_) {
      return conv_transpose_(x);
    }

    // torch::Dtype dtype = x.dtype();
    // if (dtype == torch::kBFloat16 && is_torch_version_less("2.1")) {
    //     x = x.to(torch::kFloat32);
    // }

    if (x.size(0) >= 64) {
      x = x.contiguous();
    }

    if (interpolate_) {
      torch::nn::functional::InterpolateFuncOptions opts;
      opts.mode(torch::kNearest);

      if (!output_size.empty()) {
        opts.size(output_size);
      } else {
        opts.scale_factor(std::vector<double>{2.0, 2.0});
      }

      x = torch::nn::functional::interpolate(x, opts);
    }

    // if (dtype == torch::kBFloat16 && is_torch_version_less("2.1")) {
    //     x = x.to(dtype);
    // }

    if (use_conv_) {
      x = conv_(x);
    }

    return x;
  }
  void load_state_dict(const StateDict& state_dict) {
    if (use_conv_transpose_) {
      // conv_transpose_
      const auto weight = state_dict.get_tensor("conv_transpose.weight");
      if (weight.defined()) {
        DCHECK_EQ(conv_transpose_->weight.sizes(), weight.sizes())
            << "conv_transpose weight size mismatch: expected "
            << conv_transpose_->weight.sizes() << " but got " << weight.sizes();
        conv_transpose_->weight.data().copy_(weight);
      }

      const auto bias = state_dict.get_tensor("conv_transpose.bias");
      if (bias.defined() && conv_transpose_->bias.defined()) {
        DCHECK_EQ(conv_transpose_->bias.sizes(), bias.sizes())
            << "conv_transpose bias size mismatch: expected "
            << conv_transpose_->bias.sizes() << " but got " << bias.sizes();
        conv_transpose_->bias.data().copy_(bias);
      }
    }
    if (use_conv_) {
      const auto conv_state_dict = state_dict.get_tensor("conv.weight");
      if (conv_state_dict.defined()) {
        DCHECK_EQ(conv_->weight.sizes(), conv_state_dict.sizes())
            << "conv weight size mismatch: expected " << conv_->weight.sizes()
            << " but got " << conv_state_dict.sizes();
        conv_->weight.data().copy_(conv_state_dict);
      }
      const auto conv_state_bias = state_dict.get_tensor("conv.bias");
      if (conv_state_bias.defined()) {
        DCHECK_EQ(conv_->bias.sizes(), conv_state_bias.sizes())
            << "conv bias size mismatch: expected " << conv_->bias.sizes()
            << " but got " << conv_state_bias.sizes();
        conv_->bias.data().copy_(conv_state_bias);
      }
    }
  }

 private:
  int64_t channels_;
  int64_t out_channels_;
  bool use_conv_;
  bool use_conv_transpose_;
  torch::nn::ConvTranspose2d conv_transpose_ = {nullptr};
  torch::nn::Conv2d conv_{nullptr};
  std::string name_;
  bool interpolate_;
};

TORCH_MODULE(Upsample2D);
class ResnetBlock2DImpl : public torch::nn::Module {
 public:
  ResnetBlock2DImpl(int64_t in_channels,
                    c10::optional<int64_t> out_channels = c10::nullopt,
                    bool conv_shortcut = false,
                    float dropout = 0.0f,
                    c10::optional<int64_t> temb_channels = c10::nullopt,
                    int64_t groups = 32,
                    c10::optional<int64_t> groups_out = c10::nullopt,
                    bool pre_norm = true,
                    float eps = 1e-6f,
                    const std::string& non_linearity = "swish",
                    bool skip_time_act = false,
                    const std::string& time_embedding_norm = "default",
                    const c10::optional<std::string>& kernel = c10::nullopt,
                    float output_scale_factor = 1.0f,
                    c10::optional<bool> use_in_shortcut = c10::nullopt,
                    bool conv_shortcut_bias = true,
                    c10::optional<int64_t> conv_2d_out_channels = c10::nullopt)
      : pre_norm_(pre_norm),
        in_channels_(in_channels),
        out_channels_(out_channels.value_or(in_channels)),
        use_conv_shortcut_(conv_shortcut),
        output_scale_factor_(output_scale_factor),
        time_embedding_norm_(time_embedding_norm),
        skip_time_act_(skip_time_act),
        kernel_(kernel) {
    int64_t groups_out_val = groups_out.value_or(groups);

    norm1_ = register_module(
        "norm1",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, in_channels_).eps(eps)));
    conv1_ =
        register_module("conv1",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                              in_channels_, out_channels_, 3)
                                              .stride(1)
                                              .padding(1)
                                              .bias(true)));

    if (temb_channels.has_value() && temb_channels.value() != 0) {
      int64_t time_proj_out_channels;
      if (time_embedding_norm == "default") {
        time_proj_out_channels = out_channels_;
      } else if (time_embedding_norm == "scale_shift") {
        time_proj_out_channels = 2 * out_channels_;
      } else {
        throw std::invalid_argument("Unknown time_embedding_norm: " +
                                    time_embedding_norm);
      }
      time_emb_proj_ = register_module(
          "time_emb_proj",
          DiTLinear(temb_channels.value(), time_proj_out_channels));
    }

    norm2_ =
        register_module("norm2",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 groups_out_val,  // num_groups
                                                 out_channels_)  // num_channels
                                                 .eps(eps)));

    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));

    int64_t conv2_out_channels = conv_2d_out_channels.value_or(out_channels_);
    conv2_ = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels_,
                                                   conv2_out_channels,
                                                   3  // kernel_size=3
                                                   )
                              .stride(1)
                              .padding(1)
                              .bias(true)));

    nonlinearity_ =
        register_module("nonlinearity", torch::nn::Functional(torch::silu));

    bool use_in_shortcut_val =
        use_in_shortcut.value_or(in_channels_ != conv2_out_channels);
    if (use_in_shortcut_val) {
      conv_shortcut_ = register_module(
          "conv_shortcut",
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(in_channels_, conv2_out_channels, 1)
                  .stride(1)
                  .padding(0)
                  .bias(conv_shortcut_bias)));
    }
  }

  torch::Tensor forward(const torch::Tensor& input_tensor,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor hidden_states = input_tensor;
    hidden_states = norm1_(hidden_states);
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = conv1_(hidden_states);
    torch::Tensor temb_processed = temb;
    torch::Tensor input_tensor_processed = input_tensor;
    if (time_emb_proj_) {
      if (!skip_time_act_) {
        temb_processed = nonlinearity_(temb_processed);
      }
      temb_processed = time_emb_proj_(temb_processed);
      temb_processed = temb_processed.unsqueeze(2).unsqueeze(2);
    }

    if (time_embedding_norm_ == "default") {
      if (temb_processed.defined()) {
        hidden_states = hidden_states + temb_processed;
      }
      hidden_states = norm2_(hidden_states);
    } else if (time_embedding_norm_ == "scale_shift") {
      if (!temb_processed.defined()) {
        throw std::invalid_argument(
            "temb must be defined for scale_shift time_embedding_norm");
      }
      auto chunks = torch::chunk(temb_processed, 2, 1);
      torch::Tensor time_scale = chunks[0];
      torch::Tensor time_shift = chunks[1];
      hidden_states = norm2_(hidden_states);
      hidden_states = hidden_states * (1 + time_scale) + time_shift;
    } else {
      hidden_states = norm2_(hidden_states);
    }
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = dropout_(hidden_states);
    hidden_states = conv2_(hidden_states);

    if (conv_shortcut_) {
      input_tensor_processed =
          conv_shortcut_(input_tensor_processed.contiguous());
    }
    torch::Tensor output_tensor =
        (input_tensor_processed + hidden_states) / output_scale_factor_;

    return output_tensor;
  }
  void load_state_dict(const StateDict& state_dict) {
    // conv1_
    const auto conv1_weight = state_dict.get_tensor("conv1.weight");
    if (conv1_weight.defined()) {
      DCHECK_EQ(conv1_->weight.sizes(), conv1_weight.sizes())
          << "conv1 weight size mismatch: expected " << conv1_->weight.sizes()
          << " but got " << conv1_weight.sizes();
      conv1_->weight.data().copy_(conv1_weight);
    }
    const auto conv1_bias = state_dict.get_tensor("conv1.bias");
    if (conv1_bias.defined() && conv1_->bias.defined()) {
      DCHECK_EQ(conv1_->bias.sizes(), conv1_bias.sizes())
          << "conv1 bias size mismatch: expected " << conv1_->bias.sizes()
          << " but got " << conv1_bias.sizes();
      conv1_->bias.data().copy_(conv1_bias);
    }
    // time_emb_proj_
    if (time_emb_proj_) {
      const auto time_emb_proj_weight =
          state_dict.get_tensor("time_emb_proj.weight");
      if (time_emb_proj_weight.defined()) {
        DCHECK_EQ(time_emb_proj_->weight.sizes(), time_emb_proj_weight.sizes())
            << "time_emb_proj weight size mismatch: expected "
            << time_emb_proj_->weight.sizes() << " but got "
            << time_emb_proj_weight.sizes();
        time_emb_proj_->weight.data().copy_(time_emb_proj_weight);
      }
      const auto time_emb_proj_bias =
          state_dict.get_tensor("time_emb_proj.bias");
      if (time_emb_proj_bias.defined() && time_emb_proj_->bias.defined()) {
        DCHECK_EQ(time_emb_proj_->bias.sizes(), time_emb_proj_bias.sizes())
            << "time_emb_proj bias size mismatch: expected "
            << time_emb_proj_->bias.sizes() << " but got "
            << time_emb_proj_bias.sizes();
        time_emb_proj_->bias.data().copy_(time_emb_proj_bias);
      }
    }
    // norm1_
    const auto norm1_weight = state_dict.get_tensor("norm1.weight");
    if (norm1_weight.defined()) {
      DCHECK_EQ(norm1_->weight.sizes(), norm1_weight.sizes())
          << "norm1 weight size mismatch: expected " << norm1_->weight.sizes()
          << " but got " << norm1_weight.sizes();
      norm1_->weight.data().copy_(norm1_weight);
    }
    const auto norm1_bias = state_dict.get_tensor("norm1.bias");
    if (norm1_bias.defined() && norm1_->bias.defined()) {
      DCHECK_EQ(norm1_->bias.sizes(), norm1_bias.sizes())
          << "norm1 bias size mismatch: expected " << norm1_->bias.sizes()
          << " but got " << norm1_bias.sizes();
      norm1_->bias.data().copy_(norm1_bias);
    }
    // norm2_
    const auto norm2_weight = state_dict.get_tensor("norm2.weight");
    if (norm2_weight.defined()) {
      DCHECK_EQ(norm2_->weight.sizes(), norm2_weight.sizes())
          << "norm2 weight size mismatch: expected " << norm2_->weight.sizes()
          << " but got " << norm2_weight.sizes();
      norm2_->weight.data().copy_(norm2_weight);
    }
    const auto norm2_bias = state_dict.get_tensor("norm2.bias");
    if (norm2_bias.defined() && norm2_->bias.defined()) {
      DCHECK_EQ(norm2_->bias.sizes(), norm2_bias.sizes())
          << "norm2 bias size mismatch: expected " << norm2_->bias.sizes()
          << " but got " << norm2_bias.sizes();
      norm2_->bias.data().copy_(norm2_bias);
    }
    // conv2_
    const auto conv2_weight = state_dict.get_tensor("conv2.weight");
    if (conv2_weight.defined()) {
      DCHECK_EQ(conv2_->weight.sizes(), conv2_weight.sizes())
          << "conv2 weight size mismatch: expected " << conv2_->weight.sizes()
          << " but got " << conv2_weight.sizes();
      conv2_->weight.data().copy_(conv2_weight);
    }
    const auto conv2_bias = state_dict.get_tensor("conv2.bias");
    if (conv2_bias.defined() && conv2_->bias.defined()) {
      DCHECK_EQ(conv2_->bias.sizes(), conv2_bias.sizes())
          << "conv2 bias size mismatch: expected " << conv2_->bias.sizes()
          << " but got " << conv2_bias.sizes();
      conv2_->bias.data().copy_(conv2_bias);
    }
    if (conv_shortcut_) {
      // conv_shortcut_
      const auto conv_shortcut_weight =
          state_dict.get_tensor("conv_shortcut.weight");
      if (conv_shortcut_weight.defined()) {
        DCHECK_EQ(conv_shortcut_->weight.sizes(), conv_shortcut_weight.sizes())
            << "conv_shortcut weight size mismatch: expected "
            << conv_shortcut_->weight.sizes() << " but got "
            << conv_shortcut_weight.sizes();
        conv_shortcut_->weight.data().copy_(conv_shortcut_weight);
      }
      const auto conv_shortcut_bias =
          state_dict.get_tensor("conv_shortcut.bias");
      if (conv_shortcut_bias.defined() && conv_shortcut_->bias.defined()) {
        DCHECK_EQ(conv_shortcut_->bias.sizes(), conv_shortcut_bias.sizes())
            << "conv_shortcut bias size mismatch: expected "
            << conv_shortcut_->bias.sizes() << " but got "
            << conv_shortcut_bias.sizes();
        conv_shortcut_->bias.data().copy_(conv_shortcut_bias);
      }
    }
  }

 private:
  bool pre_norm_;
  int64_t in_channels_;
  int64_t out_channels_;
  bool use_conv_shortcut_;
  float output_scale_factor_;
  std::string time_embedding_norm_;
  bool skip_time_act_;
  c10::optional<std::string> kernel_;

  torch::nn::GroupNorm norm1_{nullptr};
  torch::nn::Conv2d conv1_{nullptr};
  DiTLinear time_emb_proj_{nullptr};
  torch::nn::GroupNorm norm2_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  torch::nn::Conv2d conv2_{nullptr};
  torch::nn::Functional nonlinearity_{nullptr};
  torch::nn::Conv2d conv_shortcut_{nullptr};

  std::function<torch::Tensor(const torch::Tensor&)> upsample_;
  Upsample2D upsample_module_{nullptr};
  std::function<torch::Tensor(const torch::Tensor&)> downsample_;
  Downsample2D downsample_module_{nullptr};
};
TORCH_MODULE(ResnetBlock2D);
/**
 * Base class for all downsampling blocks in UNet architecture.
 * Defines a common interface and shared components for all downsampling blocks.
 * Each derived class must implement the forward() method.
 */
class UNetMidBlock2DImpl : public torch::nn::Module {
 public:
  UNetMidBlock2DImpl(int64_t in_channels,
                     int64_t temb_channels,
                     float dropout = 0.0f,
                     int64_t num_layers = 1,
                     float resnet_eps = 1e-6f,
                     const std::string& resnet_time_scale_shift = "default",
                     const std::string& resnet_act_fn = "swish",
                     int64_t resnet_groups = 32,
                     bool resnet_pre_norm = true,
                     bool add_attention = true,
                     int64_t attention_head_dim = 512,
                     float output_scale_factor = 1.0f) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    int64_t adjusted_resnet_groups = resnet_groups;
    if (adjusted_resnet_groups == 0) {
      adjusted_resnet_groups = std::min(in_channels / 4, int64_t(32));
    }
    int64_t attn_groups;
    if (resnet_time_scale_shift == "default") {
      attn_groups = adjusted_resnet_groups;
    }
    add_attention_ = add_attention;
    resnets_->push_back(ResnetBlock2D(
        in_channels,
        in_channels,
        false,  // conv_shortcut
        dropout,
        temb_channels,
        adjusted_resnet_groups,
        c10::nullopt,  // groups_out
        resnet_pre_norm,
        resnet_eps,
        resnet_act_fn,
        false,  // skip_time_act
        resnet_time_scale_shift,
        c10::nullopt,         // kernel
        output_scale_factor,  // output_scale_factor
        c10::nullopt,         // use_in_shortcut
        true,                 // conv_shortcut_bias
        c10::nullopt          // conv_2d_out_channels
        ));
    int64_t attn_head_dim = attention_head_dim;
    int64_t num_heads = in_channels / attn_head_dim;
    for (int64_t i = 0; i < num_layers; ++i) {
      if (add_attention_) {
        attentions_->push_back(Attention(
            in_channels,
            num_heads,
            attn_head_dim,
            output_scale_factor,  // rescale_output_factor
            resnet_eps,
            attn_groups,  // norm_num_groups
            (resnet_time_scale_shift == "spatial")
                ? std::optional<int64_t>(temb_channels)
                : std::nullopt,  // spatial_norm_dim
            true,                // residual_connection
            true,                // bias
            true,                // upcast_softmax
            true                 // _from_deprecated_attn_block
            ));
      } else {
        // Add an empty module as a placeholder.
        attentions_->push_back(torch::nn::Sequential());
      }
      resnets_->push_back(ResnetBlock2D(
          in_channels,
          in_channels,
          false,  // conv_shortcut
          dropout,
          temb_channels,
          adjusted_resnet_groups,
          c10::nullopt,  // groups_out
          resnet_pre_norm,
          resnet_eps,
          resnet_act_fn,
          false,  // skip_time_act
          resnet_time_scale_shift,
          c10::nullopt,         // kernel
          output_scale_factor,  // output_scale_factor
          c10::nullopt,         // use_in_shortcut
          true,                 // conv_shortcut_bias
          c10::nullopt          // conv_2d_out_channels
          ));
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor current_hidden =
        resnets_[0]->as<ResnetBlock2D>()->forward(hidden_states, temb);
    for (size_t i = 0; i < attentions_->size(); ++i) {
      if (add_attention_) {
        auto attn = attentions_[i]->as<Attention>();
        current_hidden = attn->forward(current_hidden, temb);
      }
      auto resnet = resnets_[i + 1]->as<ResnetBlock2D>();
      current_hidden = resnet->forward(current_hidden, temb);
    }

    return current_hidden;
  }
  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < attentions_->size(); ++i) {
      if (add_attention_) {
        attentions_[i]->as<Attention>()->load_state_dict(
            state_dict.get_dict_with_prefix("attentions." + std::to_string(i) +
                                            "."));
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList attentions_{nullptr};
  bool add_attention_;
};
TORCH_MODULE(UNetMidBlock2D);

class BaseDownEncoderBlockImpl : public torch::nn::Module {
 public:
  /**
   * Pure virtual function for the forward pass of the downsampling block.
   * All derived classes must implement this method.
   *
   * @param hidden_states Input feature tensor.
   * @param temb Optional time embedding tensor.
   * @param args Optional additional arguments (for extension).
   * @param scale Optional scaling factor for certain modules.
   *
   * @returns A tuple containing:
   *   - The output hidden states after processing.
   *   - A list of intermediate output states (for skip connections).
   */
  virtual torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb = torch::Tensor(),
      const std::vector<torch::Tensor>& args = {}) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;

 protected:
  // Protected constructor to prevent direct instantiation of the base class
  BaseDownEncoderBlockImpl() = default;

  // Common components for all downsampling blocks
  torch::nn::ModuleList resnets_{nullptr};       // List of ResNet blocks
  torch::nn::ModuleList downsamplers_{nullptr};  // List of downsamplers
};
TORCH_MODULE(BaseDownEncoderBlock);
class BaseUpDecoderBlockImpl : public torch::nn::Module {
 public:
  /**
   * Pure virtual forward method for all up blocks.
   * Derived classes must implement this to define forward propagation.
   *
   * @param hidden_states Input feature tensor.
   * @param temb Optional time embedding tensor.
   * @param skip_states Optional skip connection tensors from down blocks.
   * @param args Additional optional tensors for extension.
   * @param scale Optional scaling factor.
   *
   * @return Tuple of (output hidden states, intermediate output states).
   */
  virtual torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb = torch::Tensor(),
      const std::vector<torch::Tensor>& skip_states = {},
      const std::vector<torch::Tensor>& args = {}) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;

 protected:
  // Protected constructor to prevent direct instantiation
  BaseUpDecoderBlockImpl() = default;

  // Common components for up blocks (resnets, upsamplers, etc.)
  torch::nn::ModuleList resnets_{nullptr};     // Residual blocks
  torch::nn::ModuleList upsamplers_{nullptr};  // Upsampling layers
};
TORCH_MODULE(BaseUpDecoderBlock);

class DownEncoderBlock2DImpl : public BaseDownEncoderBlockImpl {
 public:
  DownEncoderBlock2DImpl(int64_t in_channels,
                         int64_t out_channels,
                         int64_t temb_channels,
                         float dropout = 0.0f,
                         int64_t num_layers = 1,
                         float resnet_eps = 1e-6f,
                         const std::string& resnet_time_scale_shift = "default",
                         const std::string& resnet_act_fn = "swish",
                         int64_t resnet_groups = 32,
                         bool resnet_pre_norm = true,
                         float output_scale_factor = 1.0f,
                         bool add_downsample = true,
                         int64_t downsample_padding = 1) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    downsamplers_ = register_module("downsamplers", torch::nn::ModuleList());
    // initialize resnet blocks
    for (int64_t i = 0; i < num_layers; ++i) {
      const int64_t current_in_channels = (i == 0) ? in_channels : out_channels;

      resnets_->push_back(ResnetBlock2D(current_in_channels,  // in channels
                                        out_channels,
                                        false,  // conv_shortcut
                                        dropout,
                                        temb_channels,
                                        resnet_groups,
                                        c10::nullopt,  // groups_out
                                        resnet_pre_norm,
                                        resnet_eps,
                                        resnet_act_fn,
                                        false,  // skip_time_act
                                        resnet_time_scale_shift,
                                        c10::nullopt,  // kernel
                                        output_scale_factor,
                                        c10::nullopt,  // use_in_shortcut
                                        true,          // conv_shortcut_bias
                                        c10::nullopt   // conv_2d_out_channels
                                        ));
    }

    // initialize downsamplers if needed
    if (add_downsample) {
      downsamplers_->push_back(Downsample2D(out_channels,
                                            true,  // use_conv
                                            out_channels,
                                            downsample_padding,
                                            "op"));
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& args = {}) {
    std::vector<torch::Tensor> output_states;
    torch::Tensor current_hidden = hidden_states;
    // handle resnet blocks
    for (size_t i = 0; i < resnets_->size(); ++i) {
      auto resnet = resnets_[i]->as<ResnetBlock2D>();
      current_hidden =
          resnet->forward(current_hidden.clone(),
                          temb.defined() ? temb.clone() : torch::Tensor());
      output_states.push_back(current_hidden);
    }
    // handle downsampling
    if (downsamplers_) {
      for (size_t i = 0; i < downsamplers_->size(); ++i) {
        auto downsampler = downsamplers_[i]->as<Downsample2D>();
        current_hidden = downsampler->forward(current_hidden.clone());
        output_states.push_back(current_hidden);
      }
    }
    return current_hidden;
  }
  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < downsamplers_->size(); ++i) {
      downsamplers_[i]->as<Downsample2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("downsamplers." + std::to_string(i) +
                                          "."));
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList downsamplers_{nullptr};
};

TORCH_MODULE(DownEncoderBlock2D);

class UpBlock2DImpl : public BaseUpDecoderBlockImpl {
 public:
  UpBlock2DImpl(int64_t in_channels,
                int64_t prev_output_channel,
                int64_t out_channels,
                int64_t temb_channels,
                std::optional<int64_t> resolution_idx = std::nullopt,
                float dropout = 0.0f,
                int64_t num_layers = 1,
                float resnet_eps = 1e-6f,
                const std::string& resnet_time_scale_shift = "default",
                const std::string& resnet_act_fn = "swish",
                int64_t resnet_groups = 32,
                bool resnet_pre_norm = true,
                float output_scale_factor = 1.0f,
                bool add_upsample = true) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t res_skip_channels =
          (i == num_layers - 1) ? in_channels : out_channels;
      int64_t resnet_in_channels =
          (i == 0) ? prev_output_channel : out_channels;
      int64_t block_in_channels = resnet_in_channels + res_skip_channels;
      resnets_->push_back(ResnetBlock2D(block_in_channels,  // in channels
                                        out_channels,
                                        false,  // conv_shortcut
                                        dropout,
                                        temb_channels,
                                        resnet_groups,
                                        c10::nullopt,  // groups_out
                                        resnet_pre_norm,
                                        resnet_eps,
                                        resnet_act_fn,
                                        false,  // skip_time_act
                                        resnet_time_scale_shift,
                                        c10::nullopt,  // kernel
                                        output_scale_factor,
                                        c10::nullopt,
                                        true,
                                        c10::nullopt));
    }
    if (add_upsample) {
      upsamplers_->push_back(Upsample2D(out_channels,
                                        true,
                                        false,
                                        out_channels,
                                        "conv",
                                        3,
                                        1,
                                        std::nullopt,
                                        1e-5,
                                        true,
                                        true,
                                        true));
    }
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb = torch::Tensor(),
      const std::vector<torch::Tensor>& res_hidden_states_tuple = {},
      const std::vector<torch::Tensor>& args = {}) override {
    std::vector<torch::Tensor> res_hidden_states_vec = res_hidden_states_tuple;

    torch::Tensor current_hidden = hidden_states;

    for (size_t i = 0; i < resnets_->size(); ++i) {
      if (res_hidden_states_vec.empty()) {
        throw std::runtime_error(
            "res_hidden_states_tuple is empty, but required for skip "
            "connections");
      }
      torch::Tensor res_hidden = res_hidden_states_vec.back();
      res_hidden_states_vec.pop_back();

      torch::Tensor concat_hidden = torch::cat({current_hidden, res_hidden}, 1);

      auto* resnet = resnets_->children()[i]->as<ResnetBlock2D>();
      current_hidden = resnet->forward(concat_hidden, temb);
    }
    if (!upsamplers_->is_empty()) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        auto* upsampler = upsamplers_->children()[i]->as<Upsample2D>();
        current_hidden = upsampler->forward(current_hidden);
      }
    }

    return current_hidden;
  }
  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < upsamplers_->size(); ++i) {
      upsamplers_[i]->as<Upsample2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("upsamplers." + std::to_string(i) +
                                          "."));
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
};
TORCH_MODULE(UpBlock2D);

class UpDecoderBlock2DImpl : public BaseUpDecoderBlockImpl {
 public:
  UpDecoderBlock2DImpl(int64_t in_channels,
                       int64_t out_channels,
                       std::optional<int64_t> resolution_idx = std::nullopt,
                       float dropout = 0.0f,
                       int64_t num_layers = 1,
                       float resnet_eps = 1e-6f,
                       const std::string& resnet_time_scale_shift = "default",
                       const std::string& resnet_act_fn = "swish",
                       int64_t resnet_groups = 32,
                       bool resnet_pre_norm = true,
                       float output_scale_factor = 1.0f,
                       bool add_upsample = true,
                       std::optional<int64_t> temb_channels = std::nullopt) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());

    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t input_channels = (i == 0) ? in_channels : out_channels;
      resnets_->push_back(ResnetBlock2D(
          input_channels,
          out_channels,
          false,  // conv_shortcut
          dropout,
          temb_channels.has_value() ? temb_channels.value() : 0,
          resnet_groups,
          c10::nullopt,  // groups_out
          resnet_pre_norm,
          resnet_eps,
          resnet_act_fn,
          false,  // skip_time_act
          resnet_time_scale_shift,
          c10::nullopt,  // kernel
          output_scale_factor,
          c10::nullopt,  // use_in_shortcut
          true,          // conv_shortcut_bias
          c10::nullopt   // conv_2d_out_channels
          ));
    }

    if (add_upsample) {
      add_upsample_ = true;
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsamplers_->push_back(Upsample2D(out_channels,
                                        true,
                                        false,
                                        out_channels,
                                        "conv",
                                        3,
                                        1,
                                        std::nullopt,
                                        1e-5,
                                        true,
                                        true,
                                        true));
    }

    resolution_idx_ = resolution_idx;
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& skip_states = {},
                        const std::vector<torch::Tensor>& args = {}) {
    torch::Tensor current_hidden = hidden_states;

    for (size_t i = 0; i < resnets_->size(); ++i) {
      current_hidden =
          resnets_[i]->as<ResnetBlock2D>()->forward(current_hidden, temb);
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        auto* upsampler = upsamplers_->children()[i]->as<Upsample2D>();
        current_hidden = upsampler->forward(current_hidden);
      }
    }
    return current_hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }

    if (add_upsample_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        upsamplers_[i]->as<Upsample2D>()->load_state_dict(
            state_dict.get_dict_with_prefix("upsamplers." + std::to_string(i) +
                                            "."));
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
  bool add_upsample_ = false;
  std::optional<int64_t> resolution_idx_;
};
TORCH_MODULE(UpDecoderBlock2D);

inline std::shared_ptr<BaseDownEncoderBlockImpl> get_down_block(
    const std::string& down_block_type,
    int num_layers,
    int in_channels,
    int out_channels,
    int temb_channels,
    bool add_downsample,
    float resnet_eps,
    const std::string& resnet_act_fn,
    int transformer_layers_per_block = 1,
    std::optional<int> num_attention_heads = std::nullopt,
    std::optional<int> resnet_groups = std::nullopt,
    std::optional<int> cross_attention_dim = std::nullopt,
    std::optional<int> downsample_padding = std::nullopt,
    bool dual_cross_attention = false,
    bool use_linear_projection = false,
    bool only_cross_attention = false,
    bool upcast_attention = false,
    const std::string& resnet_time_scale_shift = "default",
    const std::string& attention_type = "default",
    bool resnet_skip_time_act = false,
    float resnet_out_scale_factor = 1.0f,
    std::optional<std::string> cross_attention_norm = std::nullopt,
    std::optional<int> attention_head_dim = std::nullopt,
    std::optional<std::string> downsample_type = std::nullopt,
    float dropout = 0.0f) {
  if (!attention_head_dim.has_value()) {
    std::cerr
        << "Warning: It is recommended to provide `attention_head_dim` when "
           "calling `get_down_block`. Defaulting `attention_head_dim` to "
        << num_attention_heads.value_or(0) << "." << std::endl;
    attention_head_dim = num_attention_heads;
  }

  std::string processed_block_type = down_block_type;
  if (processed_block_type.size() >= 7 &&
      processed_block_type.substr(0, 7) == "UNetRes") {
    processed_block_type = processed_block_type.substr(7);
  }

  if (processed_block_type == "DownEncoderBlock2D") {
    bool is_downsample = add_downsample;
    return std::make_shared<DownEncoderBlock2DImpl>(
        static_cast<int64_t>(in_channels),
        static_cast<int64_t>(out_channels),
        static_cast<int64_t>(temb_channels),
        static_cast<float>(dropout),
        static_cast<int64_t>(num_layers),
        static_cast<float>(resnet_eps),
        resnet_time_scale_shift,
        resnet_act_fn,
        resnet_groups.has_value() ? resnet_groups.value() : 32,  //
        true,  // resnet_pre_norm
        static_cast<float>(resnet_out_scale_factor),
        is_downsample,  // add_downsample
        downsample_padding.has_value()
            ? static_cast<int64_t>(downsample_padding.value())
            : 1);
  } else {
    throw std::invalid_argument(processed_block_type + " does not exist.");
  }
}

/**
 * Creates and returns an upsampling block based on the specified type.
 */
inline std::shared_ptr<BaseUpDecoderBlockImpl> get_up_block(
    const std::string& up_block_type,
    int num_layers,
    int in_channels,
    int out_channels,
    int prev_output_channel,
    int temb_channels,
    bool add_upsample,
    float resnet_eps,
    const std::string& resnet_act_fn,
    std::optional<int> resolution_idx = std::nullopt,
    int transformer_layers_per_block = 1,
    std::optional<int> num_attention_heads = std::nullopt,
    std::optional<int> resnet_groups = std::nullopt,
    std::optional<int> cross_attention_dim = std::nullopt,
    bool dual_cross_attention = false,
    bool use_linear_projection = false,
    bool only_cross_attention = false,
    bool upcast_attention = false,
    const std::string& resnet_time_scale_shift = "default",
    const std::string& attention_type = "default",
    bool resnet_skip_time_act = false,
    float resnet_out_scale_factor = 1.0f,
    std::optional<std::string> cross_attention_norm = std::nullopt,
    std::optional<int> attention_head_dim = std::nullopt,
    std::optional<std::string> upsample_type = std::nullopt,
    float dropout = 0.0f) {
  // Handle default for attention_head_dim
  if (!attention_head_dim.has_value()) {
    std::cerr
        << "Warning: It is recommended to provide `attention_head_dim` when "
           "calling `get_up_block`. Defaulting `attention_head_dim` to "
        << num_attention_heads.value_or(0) << "." << std::endl;
    attention_head_dim = num_attention_heads;
  }

  // Process up_block_type: remove "UNetRes" prefix if present
  std::string processed_block_type = up_block_type;
  if (processed_block_type.size() >= 7 &&
      processed_block_type.substr(0, 7) == "UNetRes") {
    processed_block_type = processed_block_type.substr(7);
  }

  // Create corresponding up block based on processed type
  if (processed_block_type == "UpDecoderBlock2D") {
    return std::make_shared<UpDecoderBlock2DImpl>(
        static_cast<int64_t>(in_channels),
        static_cast<int64_t>(out_channels),
        resolution_idx.has_value()
            ? std::optional<int64_t>(resolution_idx.value())
            : std::nullopt,
        static_cast<float>(dropout),
        static_cast<int64_t>(num_layers),
        static_cast<float>(resnet_eps),
        resnet_time_scale_shift,
        resnet_act_fn,
        resnet_groups.has_value() ? resnet_groups.value() : 32,  //
        true,  // resnet_pre_norm
        resnet_out_scale_factor,
        add_upsample,
        static_cast<int64_t>(temb_channels));
  } else {
    throw std::invalid_argument(processed_block_type + " does not exist.");
  }
}
// Diagonal Gaussian distribution for VAE latent space
// This class is used to represent the distribution of the latent space in VAE
// models. It provides methods for sampling, calculating KL divergence, and
// negative log likelihood.
class DiagonalGaussianDistribution {
 public:
  DiagonalGaussianDistribution(torch::Tensor parameters,
                               bool deterministic = false)
      : parameters_(std::move(parameters)), deterministic_(deterministic) {
    auto chunks = parameters_.chunk(2, 1);
    mean_ = chunks[0];
    logvar_ = chunks[1];

    logvar_ = torch::clamp(logvar_, -30.0f, 20.0f);

    std_ = torch::exp(0.5f * logvar_);
    var_ = torch::exp(logvar_);

    if (deterministic_) {
      std_.fill_(0.0f);
      var_.fill_(0.0f);
    }
  }

  torch::Tensor sample() const {
    torch::TensorOptions options = mean_.options();
    return mean_ + std_ * torch::randn_like(mean_, options);
  }

  torch::Tensor kl(const std::optional<DiagonalGaussianDistribution>& other =
                       std::nullopt) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }

    if (!other.has_value()) {
      return 0.5f * torch::sum(torch::pow(mean_, 2) + var_ - 1.0f - logvar_,
                               {1, 2, 3});
    } else {
      const auto& other_dist = other.value();
      return 0.5f * torch::sum(torch::pow(mean_ - other_dist.mean_, 2) /
                                       other_dist.var_ +
                                   var_ / other_dist.var_ - 1.0f - logvar_ +
                                   other_dist.logvar_,
                               {1, 2, 3});
    }
  }

  torch::Tensor nll(const torch::Tensor& sample,
                    const std::vector<int64_t>& dims = {1, 2, 3}) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }
    const float logtwopi = std::log(2.0f * M_PI);
    return 0.5f *
           torch::sum(logtwopi + logvar_ + torch::pow(sample - mean_, 2) / var_,
                      dims);
  }

  torch::Tensor mode() const { return mean_; }

  const torch::Tensor& mean() const { return mean_; }
  const torch::Tensor& std() const { return std_; }
  const torch::Tensor& var() const { return var_; }
  const torch::Tensor& logvar() const { return logvar_; }

 private:
  torch::Tensor parameters_;
  torch::Tensor mean_;
  torch::Tensor logvar_;
  torch::Tensor std_;
  torch::Tensor var_;
  bool deterministic_;
};

struct AutoencoderKLOutput {
  DiagonalGaussianDistribution latent_dist;
  explicit AutoencoderKLOutput(DiagonalGaussianDistribution latent_dist)
      : latent_dist(std::move(latent_dist)) {}
};
struct DecoderOutput {
  torch::Tensor sample;
  torch::Tensor commit_loss;
  explicit DecoderOutput(torch::Tensor sample, torch::Tensor commit_loss = {})
      : sample(std::move(sample)), commit_loss(std::move(commit_loss)) {}
};

// VAE standard encoder implementation
// This class is used to encode images into latent representations.
class VAEEncoderImpl : public torch::nn::Module {
 public:
  VAEEncoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.vae_in_channels(), args.vae_block_out_channels()[0], 3)
                .stride(1)
                .padding(1)
                .bias(true)));
    // downblocks
    // TODO
    int32_t output_channels = args.vae_block_out_channels()[0];
    for (size_t i = 0; i < args.vae_down_block_types().size(); i++) {
      int32_t input_channels = output_channels;
      output_channels = args.vae_block_out_channels()[i];
      bool is_final_block = (i == args.vae_block_out_channels().size() - 1);
      auto down_block =
          get_down_block(args.vae_down_block_types()[i],
                         args.vae_layers_per_block(),
                         input_channels,
                         output_channels,
                         0,  // temb_channels, using 0 instead of nullptr
                         !is_final_block,
                         1e-6,
                         args.vae_act_fn(),
                         1,
                         std::nullopt,
                         args.vae_norm_num_groups(),
                         std::nullopt,  // cross_attention_dim
                         0,
                         false,            // dual_cross_attention
                         false,            // use_linear_projection
                         false,            // only_cross_attention
                         false,            // upcast_attention
                         "default",        // resnet_time_scale_shift
                         "default",        // attention_type
                         false,            // resnet_skip_time_act
                         1.0f,             // resnet_out_scale_factor
                         std::nullopt,     // cross_attention_norm
                         output_channels,  // attention_head_dim
                         std::nullopt,     // downsample_type
                         0.0f              // dropout
          );
      down_blocks_->push_back(down_block);
    }
    // mid blocks
    // TODO
    mid_block_ =
        register_module("mid_block",
                        UNetMidBlock2D(args.vae_block_out_channels().back(),
                                       0,
                                       0.0f,
                                       1,
                                       1e-6f,
                                       "default",
                                       args.vae_act_fn(),
                                       args.vae_norm_num_groups(),
                                       true,
                                       args.vae_mid_block_add_attention(),
                                       args.vae_block_out_channels().back(),
                                       1.0f));
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(args.vae_norm_num_groups(),
                                        args.vae_block_out_channels().back())
                .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(args.vae_block_out_channels().back(),
                                     2 * args.vae_latent_channels(),
                                     3)
                .padding(1)
                .bias(true)));
  }
  torch::Tensor forward(const torch::Tensor& images) {
    auto sample = conv_in_(images);
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      auto down_block = down_blocks_[i]->as<BaseDownEncoderBlock>();
      sample = down_block->forward(sample);
    }
    sample = mid_block_(sample);

    sample = conv_norm_out_(sample);

    sample = conv_act_(sample);

    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEEecoder";
    // conv_in_
    const auto conv_in_weight = state_dict.get_tensor("conv_in.weight");
    if (conv_in_weight.defined()) {
      DCHECK_EQ(conv_in_->weight.sizes(), conv_in_weight.sizes())
          << "conv_in weight size mismatch";
      conv_in_->weight.data().copy_(conv_in_weight);
    }
    const auto conv_in_bias = state_dict.get_tensor("conv_in.bias");
    if (conv_in_bias.defined() && conv_in_->bias.defined()) {
      DCHECK_EQ(conv_in_->bias.sizes(), conv_in_bias.sizes())
          << "conv_in bias size mismatch";
    }

    // conv_norm_out_
    const auto conv_norm_out_weight =
        state_dict.get_tensor("conv_norm_out.weight");
    if (conv_norm_out_weight.defined()) {
      DCHECK_EQ(conv_norm_out_->weight.sizes(), conv_norm_out_weight.sizes())
          << "conv_norm_out weight size mismatch";
      conv_norm_out_->weight.data().copy_(conv_norm_out_weight);
    }
    const auto conv_norm_out_bias = state_dict.get_tensor("conv_norm_out.bias");
    if (conv_norm_out_bias.defined() && conv_norm_out_->bias.defined()) {
      DCHECK_EQ(conv_norm_out_->bias.sizes(), conv_norm_out_bias.sizes())
          << "conv_norm_out bias size mismatch";
      conv_norm_out_->bias.data().copy_(conv_norm_out_bias);
    }
    // conv_out_
    const auto conv_out_weight = state_dict.get_tensor("conv_out.weight");
    if (conv_out_weight.defined()) {
      DCHECK_EQ(conv_out_->weight.sizes(), conv_out_weight.sizes())
          << "conv_out weight size mismatch";
      conv_out_->weight.data().copy_(conv_out_weight);
    }
    const auto conv_out_bias = state_dict.get_tensor("conv_out.bias");
    if (conv_out_bias.defined() && conv_out_->bias.defined()) {
      DCHECK_EQ(conv_out_->bias.sizes(), conv_out_bias.sizes())
          << "conv_out bias size mismatch";
      conv_out_->bias.data().copy_(conv_out_bias);
    }
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      down_blocks_[i]->as<BaseDownEncoderBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." + std::to_string(i) +
                                          "."));
    }
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  UNetMidBlock2D mid_block_{nullptr};
};
TORCH_MODULE(VAEEncoder);
// VAE standart decoder implementation
//  This class is used to decode the latent representations into images.
class VAEDecoderImpl : public torch::nn::Module {
 public:
  VAEDecoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(args.vae_latent_channels(),
                                     args.vae_block_out_channels().back(),
                                     3)
                .stride(1)
                .padding(1)
                .bias(true)));
    // mid blocks
    // TODO
    mid_block_ =
        register_module("mid_block",
                        UNetMidBlock2D(args.vae_block_out_channels().back(),
                                       0,
                                       0.0f,
                                       1,
                                       1e-6f,
                                       "default",
                                       args.vae_act_fn(),
                                       args.vae_norm_num_groups(),
                                       true,
                                       args.vae_mid_block_add_attention(),
                                       args.vae_block_out_channels().back(),
                                       1.0f));
    // up blocks
    std::vector<int64_t> reversed_block_out_channels(
        args.vae_block_out_channels().rbegin(),
        args.vae_block_out_channels().rend());
    int64_t output_channel = reversed_block_out_channels[0];

    for (size_t i = 0; i < args.vae_up_block_types().size(); ++i) {
      const std::string& up_block_type = args.vae_up_block_types()[i];
      int64_t prev_output_channel = output_channel;
      output_channel = reversed_block_out_channels[i];
      bool is_final_block = (i == args.vae_block_out_channels().size() - 1);
      // Create the up block using the factory function
      auto up_block = get_up_block(up_block_type,
                                   args.vae_layers_per_block() + 1,
                                   prev_output_channel,
                                   output_channel,
                                   prev_output_channel,
                                   0,  // temb_channels
                                   !is_final_block,
                                   1e-6,
                                   args.vae_act_fn(),
                                   std::nullopt,  // resolution_idx
                                   1,  // transformer_layers_per_block
                                   std::nullopt,  // num_attention_heads
                                   args.vae_norm_num_groups(),  // resnet_groups
                                   std::nullopt,    // cross_attention_dim
                                   false,           // dual_cross_attention
                                   false,           // use_linear_projection
                                   false,           // only_cross_attention
                                   false,           // upcast_attention
                                   "group",         // resnet_time_scale_shift
                                   "default",       // attention_type
                                   false,           // resnet_skip_time_act
                                   1.0f,            // resnet_out_scale_factor
                                   std::nullopt,    // cross_attention_norm
                                   output_channel,  // attention_head_dim
                                   std::nullopt,    // upsample_type
                                   0.0f             // dropout
      );
      up_blocks_->push_back(
          register_module("up_block_" + std::to_string(i), up_block));
      prev_output_channel = output_channel;
    }

    // GroupNorm：num_channels=block_out_channels[0], num_groups=norm_num_groups
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(
                args.vae_norm_num_groups(),       // num_groups
                args.vae_block_out_channels()[0]  // num_channels
                )
                .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.vae_block_out_channels()[0], args.vae_out_channels(), 3)
                .padding(1)
                .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& latents) {
    auto sample = conv_in_(latents);
    sample = mid_block_(sample);
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      sample = up_blocks_[i]->as<BaseUpDecoderBlock>()->forward(sample);
    }
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);
    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEDecoder";
    // conv_in_
    const auto conv_in_weight = state_dict.get_tensor("conv_in.weight");
    if (conv_in_weight.defined()) {
      DCHECK_EQ(conv_in_->weight.sizes(), conv_in_weight.sizes())
          << "conv_in weight size mismatch";
      conv_in_->weight.data().copy_(conv_in_weight);
    }
    const auto conv_in_bias = state_dict.get_tensor("conv_in.bias");
    if (conv_in_bias.defined() && conv_in_->bias.defined()) {
      DCHECK_EQ(conv_in_->bias.sizes(), conv_in_bias.sizes())
          << "conv_in bias size mismatch";
      conv_in_->bias.data().copy_(conv_in_bias);
    }
    // mid_block_
    //  mid_block_ is a UNetMidBlock2D, so we load its state
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      up_blocks_[i]->as<BaseUpDecoderBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("up_blocks." + std::to_string(i) +
                                          "."));
    }
    // conv_norm_out_
    const auto conv_norm_out_weight =
        state_dict.get_tensor("conv_norm_out.weight");
    if (conv_norm_out_weight.defined()) {
      DCHECK_EQ(conv_norm_out_->weight.sizes(), conv_norm_out_weight.sizes())
          << "conv_norm_out weight size mismatch";
      conv_norm_out_->weight.data().copy_(conv_norm_out_weight);
    }
    const auto conv_norm_out_bias = state_dict.get_tensor("conv_norm_out.bias");
    if (conv_norm_out_bias.defined() && conv_norm_out_->bias.defined()) {
      DCHECK_EQ(conv_norm_out_->bias.sizes(), conv_norm_out_bias.sizes())
          << "conv_norm_out bias size mismatch";
      conv_norm_out_->bias.data().copy_(conv_norm_out_bias);
    }
    // conv_out_
    const auto conv_out_weight = state_dict.get_tensor("conv_out.weight");
    if (conv_out_weight.defined()) {
      DCHECK_EQ(conv_out_->weight.sizes(), conv_out_weight.sizes())
          << "conv_out weight size mismatch";
      conv_out_->weight.data().copy_(conv_out_weight);
    }

    const auto conv_out_bias = state_dict.get_tensor("conv_out.bias");
    if (conv_out_bias.defined() && conv_out_->bias.defined()) {
      DCHECK_EQ(conv_out_->bias.sizes(), conv_out_bias.sizes())
          << "conv_out bias size mismatch";
      conv_out_->bias.data().copy_(conv_out_bias);
    }
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  UNetMidBlock2D mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_;
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};
};
TORCH_MODULE(VAEDecoder);
// VAE implementation, including encoder and decoder
class VAEImpl : public torch::nn::Module {
 public:
  VAEImpl(const ModelContext& context,
          torch::Device device,
          torch::ScalarType dtype)
      : args_(context.get_model_args()), device_(device), dtype_(dtype) {
    encoder_ = register_module("encoder", VAEEncoder(context));
    decoder_ = register_module("decoder", VAEDecoder(context));
    if (args_.vae_use_quant_conv()) {
      quant_conv_ = register_module("quant_conv",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                        2 * args_.vae_latent_channels(),
                                        2 * args_.vae_latent_channels(),
                                        1)));
    }
    if (args_.vae_use_post_quant_conv()) {
      post_quant_conv_ = register_module(
          "post_quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              args_.vae_latent_channels(), args_.vae_latent_channels(), 1)));
    }
    encoder_->to(dtype_);
    decoder_->to(dtype_);
    if (args_.vae_use_quant_conv()) {
      quant_conv_->to(dtype_);
    }
    if (args_.vae_use_post_quant_conv()) {
      post_quant_conv_->to(dtype_);
    }
    // tile_sample_min_size_ = args.sample_size();
    // tile_latent_min_size_ = int32_t(tile_sample_min_size_ / (2 ^
    // (args.block_out_channels().size() - 1)));
  }
  // Enable tiled VAE decoding. When this option is enabled, the VAE will split
  // the input tensor into tiles to
  //     compute decoding and encoding in several steps. This is useful for
  //     saving a large amount of memory and to allow processing larger images.
  void enable_slicing(bool enable) { use_slicing_ = enable; }
  void disable_slicing() { use_slicing_ = false; }
  // Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this
  // method will go back to computing
  //     decoding in one step.
  //  void enable_tiling(bool enable) {
  //      use_tiling_ = enable;
  //  }
  //  void disable_tiling() {
  //      use_tiling_ = false;
  //  }

  // Encode a batch of images into latent representations.
  torch::Tensor encode_(const torch::Tensor& images) {
    auto enc = encoder_(images);
    if (args_.vae_use_quant_conv()) {
      enc = quant_conv_(enc);
    }
    return enc;
  }
  AutoencoderKLOutput encode(const torch::Tensor& images) {
    torch::Tensor hidden_states;
    if (use_slicing_) {
      std::vector<torch::Tensor> latent_slices;
      for (const auto& x_slice : images.split(1)) {
        latent_slices.push_back(encode_(x_slice));
      }
      hidden_states = torch::cat(latent_slices, 0);
    } else {
      hidden_states = encode_(images);
    }
    auto posterior = DiagonalGaussianDistribution(hidden_states);
    return AutoencoderKLOutput(posterior);
  }

  // Decode a batch of latent representations into images.
  DecoderOutput decode_(const torch::Tensor& latents) {
    torch::Tensor processed_latents = latents;

    if (args_.vae_use_post_quant_conv()) {
      processed_latents = post_quant_conv_(processed_latents);
    }

    auto dec = decoder_(processed_latents);
    return DecoderOutput(dec);
  }
  DecoderOutput decode(
      const torch::Tensor& latents,
      const std::optional<torch::Generator>& generator = std::nullopt) {
    torch::Tensor images;
    if (use_slicing_ && latents.size(0) > 1) {
      std::vector<torch::Tensor> image_slices;
      for (const auto& latent_slice : latents.split(1)) {
        image_slices.push_back(decode_(latent_slice).sample);
      }
      images = torch::cat(image_slices, 0);
    } else {
      images = decode_(latents).sample;
    }
    return DecoderOutput(images);
  }
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    int64_t seed = 42;
    int batch_size = 1;
    int num_channels_latents = 16;
    int height = 180;
    int width = 180;
    torch::manual_seed(seed);
    torch::Tensor sample =
        torch::randn({batch_size, num_channels_latents, height, width},
                     torch::dtype(torch::kFloat32).device(device_));
    DecoderOutput output = decode_(sample);
    auto image_processor = VAEImageProcessor(true, 16);
    auto image = image_processor->postprocess(output.sample);
    return torch::Tensor();
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
  // TODO: Implement the forward method

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
          post_quant_conv_->weight.data().copy_(weight);
          is_post_quant_conv_loaded = true;
        }

        const auto bias = state_dict->get_tensor("post_quant_conv.bias");
        if (bias.defined() && post_quant_conv_->bias.defined()) {
          post_quant_conv_->bias.data().copy_(bias);
        }
      }
    }
    LOG(INFO) << "VAE model loaded successfully.";
  }

 private:
  bool is_quant_conv_loaded{false};
  bool is_post_quant_conv_loaded{false};
  VAEEncoder encoder_{nullptr};
  VAEDecoder decoder_{nullptr};
  torch::nn::Conv2d quant_conv_{nullptr};
  torch::nn::Conv2d post_quant_conv_{nullptr};
  bool use_post_quant_conv_{false};
  bool use_slicing_{false};
  // WordEmbedding embedding_{nullptr};
  ModelArgs args_;
  torch::Device device_;
  torch::ScalarType dtype_;
};
TORCH_MODULE(VAE);
// register the VAE model with the model registry
REGISTER_MODEL_ARGS(AutoencoderKL, [&] {
  LOAD_ARG_OR(vae_in_channels, "in_channels", 3);
  LOAD_ARG_OR(vae_out_channels, "out_channels", 3);
  LOAD_ARG_OR(vae_down_block_types,
              "down_block_types",
              (std::vector<std::string>{"DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D"}));
  LOAD_ARG_OR(vae_up_block_types,
              "up_block_types",
              (std::vector<std::string>{"UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D"}));
  LOAD_ARG_OR(vae_block_out_channels,
              "block_out_channels",
              (std::vector<int64_t>{128, 256, 512, 512}));
  LOAD_ARG_OR(vae_layers_per_block, "layers_per_block", 2);
  LOAD_ARG_OR(vae_act_fn, "act_fn", "silu");
  LOAD_ARG_OR(vae_latent_channels, "latent_channels", 16);
  LOAD_ARG_OR(vae_norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(vae_sample_size, "sample_size", 1024);
  LOAD_ARG_OR(vae_scale_factor, "scale_factor", 0.3611f);
  LOAD_ARG_OR(vae_shift_factor, "shift_factor", 0.1159f);
  LOAD_ARG_OR(vae_mid_block_add_attention, "mid_block_add_attention", true);
  LOAD_ARG_OR(vae_force_upcast, "force_upcast", true);
  LOAD_ARG_OR(vae_use_quant_conv, "use_quant_conv", false);
  LOAD_ARG_OR(vae_use_post_quant_conv, "use_post_quant_conv", false);
});
}  // namespace xllm