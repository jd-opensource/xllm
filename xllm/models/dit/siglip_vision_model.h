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

#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "models/model_registry.h"
#include "xllm/core/layers/common/add_matmul.h"

namespace xllm {
// siglip_vision_model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/siglip
class SiglipVisionEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionEmbeddingsImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_dim_ = model_args.mm_hidden_size();
    image_size_ = model_args.mm_image_size();
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(model_args.mm_num_channels(),
                                                   embed_dim_,
                                                   model_args.mm_patch_size())
                              .stride(model_args.mm_patch_size())
                              .bias(true)));
    patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));
    patch_embedding_->bias.set_data(patch_embedding_->bias.to(options));

    auto num_patches =
        (model_args.mm_image_size() / model_args.mm_patch_size()) *
        (model_args.mm_image_size() / model_args.mm_patch_size());
    auto num_positions = num_patches;
    position_embedding_ =
        register_parameter("position_embedding",
                           torch::randn({num_positions, embed_dim_}, options));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    auto patch_embeds =
        patch_embedding_->forward(pixel_values).flatten(2).transpose(1, 2);
    torch::Tensor embeddings =
        patch_embeds + position_embedding_.index({position_ids_});
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto pos = state_dict.get_tensor("position_embedding.weight");
    if (pos.defined()) {
      CHECK_EQ(pos.sizes(), position_embedding_.sizes())
          << "position_embedding weight size mismatch for " << name();
      position_embedding_.data().copy_(pos);
      is_position_embedding_loaded_ = true;
    }

    const auto weight = state_dict.get_tensor("patch_embedding.weight");
    if (weight.defined()) {
      CHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
          << "patch_embedding weight size mismatch for " << name();
      patch_embedding_->weight.data().copy_(weight);
      is_patch_embedding_weight_loaded_ = true;
    }

    const auto bias = state_dict.get_tensor("patch_embedding.bias");
    if (bias.defined()) {
      CHECK_EQ(patch_embedding_->bias.sizes(), bias.sizes())
          << "patch_embedding bias size mismatch for " << name();
      patch_embedding_->bias.data().copy_(bias);
      is_patch_embedding_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_position_embedding_loaded_)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_patch_embedding_weight_loaded_)
        << "weight is not loaded for " << prefix + "patch_embedding.weight";
    CHECK(is_patch_embedding_bias_loaded_)
        << "bias is not loaded for " << prefix + "patch_embedding.bias";
  }

 private:
  int64_t embed_dim_;
  int64_t image_size_;
  bool is_position_embedding_loaded_ = false;
  bool is_patch_embedding_weight_loaded_ = false;
  bool is_patch_embedding_bias_loaded_ = false;
  torch::Tensor position_ids_;
  torch::nn::Conv2d patch_embedding_ = nullptr;
  torch::Tensor position_embedding_;
};
TORCH_MODULE(SiglipVisionEmbeddings);

class SiglipAttentionImpl : public torch::nn::Module {
 public:
  SiglipAttentionImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    CHECK(model_args.mm_hidden_size() % model_args.mm_num_attention_heads() ==
          0);
    embed_dim_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    head_dim_ = embed_dim_ / num_heads_;

    q_proj_ = register_module("q_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               model_args.mm_hidden_size(),
                                               true,
                                               options));
    k_proj_ = register_module("k_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               model_args.mm_hidden_size(),
                                               true,
                                               options));
    v_proj_ = register_module("v_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               model_args.mm_hidden_size(),
                                               true,
                                               options));
    o_proj_ = register_module("o_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               model_args.mm_hidden_size(),
                                               true,
                                               options));

    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto bsz = hidden_states.size(0);
    auto tgt_len = hidden_states.size(1);

    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    query_states = shape(query_states, tgt_len, bsz);
    key_states = shape(key_states, -1, bsz);
    value_states = shape(value_states, -1, bsz);

    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query_states, key_states, value_states, torch::nullopt, 0.0, false);

    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output =
        attn_output.view(torch::IntArrayRef({bsz, tgt_len, embed_dim_}));

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
    k_proj_->load_state_dict(state_dict.get_dict_with_prefix("k_proj."));
    v_proj_->load_state_dict(state_dict.get_dict_with_prefix("v_proj."));
    o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    q_proj_->verify_loaded_weights(prefix + "q_proj.");
    k_proj_->verify_loaded_weights(prefix + "k_proj.");
    v_proj_->verify_loaded_weights(prefix + "v_proj.");
    o_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  torch::Tensor shape(torch::Tensor tensor, int64_t seq_len, int64_t bsz) {
    return tensor.view({bsz, seq_len, num_heads_, head_dim_})
        .transpose(1, 2)
        .contiguous();
  }

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;

  layer::AddMatmul o_proj_ = nullptr;
  layer::AddMatmul q_proj_ = nullptr;
  layer::AddMatmul k_proj_ = nullptr;
  layer::AddMatmul v_proj_ = nullptr;
};
TORCH_MODULE(SiglipAttention);

class SiglipMLPImpl : public torch::nn::Module {
 public:
  explicit SiglipMLPImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    act_ = register_module(
        "act",
        torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
            [](const at::Tensor& x) { return torch::gelu(x, "tanh"); })));
    fc1_ = register_module("fc1",
                           layer::AddMatmul(model_args.mm_hidden_size(),
                                            model_args.mm_intermediate_size(),
                                            true,
                                            options));
    fc2_ = register_module("fc2",
                           layer::AddMatmul(model_args.mm_intermediate_size(),
                                            model_args.mm_hidden_size(),
                                            true,
                                            options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    fc1_->load_state_dict(state_dict.get_dict_with_prefix("fc1."));
    fc2_->load_state_dict(state_dict.get_dict_with_prefix("fc2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fc1_->verify_loaded_weights(prefix + "fc1.");
    fc2_->verify_loaded_weights(prefix + "fc2.");
  }

 private:
  torch::nn::Functional act_ = nullptr;
  layer::AddMatmul fc1_ = nullptr;
  layer::AddMatmul fc2_ = nullptr;
};
TORCH_MODULE(SiglipMLP);

class SiglipEncoderLayerImpl : public torch::nn::Module {
 public:
  explicit SiglipEncoderLayerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attn_ = register_module("self_attn", SiglipAttention(context));
    layer_norm1_ = register_module(
        "layer_norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm2_ = register_module(
        "layer_norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
    layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
    layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
    layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
    mlp_ = register_module("mlp", SiglipMLP(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto residual = hidden_states;
    auto ln1_out = layer_norm1_->forward(hidden_states);
    auto h = self_attn_->forward(ln1_out) + residual;
    residual = h;
    h = layer_norm2_->forward(h);
    h = mlp_->forward(h);
    h += residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    weight::load_weight(state_dict,
                        "layer_norm1.weight",
                        layer_norm1_->weight,
                        layer_norm1_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm1.bias",
                        layer_norm1_->bias,
                        layer_norm1_bias_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.weight",
                        layer_norm2_->weight,
                        layer_norm2_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.bias",
                        layer_norm2_->bias,
                        layer_norm2_bias_loaded_);
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    CHECK(layer_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.weight";
    CHECK(layer_norm1_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.bias";
    CHECK(layer_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.weight";
    CHECK(layer_norm2_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.bias";
  }

 private:
  bool layer_norm1_weight_loaded_ = false;
  bool layer_norm1_bias_loaded_ = false;
  bool layer_norm2_weight_loaded_ = false;
  bool layer_norm2_bias_loaded_ = false;

  torch::nn::LayerNorm layer_norm1_ = nullptr;
  torch::nn::LayerNorm layer_norm2_ = nullptr;
  SiglipAttention self_attn_ = nullptr;
  SiglipMLP mlp_ = nullptr;
};
TORCH_MODULE(SiglipEncoderLayer);

class SiglipEncoderImpl : public torch::nn::Module {
 public:
  explicit SiglipEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layers_.reserve(model_args.mm_num_hidden_layers());
    for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); i++) {
      auto block = SiglipEncoderLayer(context);
      layers_.push_back(block);
    }
  }

  torch::Tensor forward(const torch::Tensor& embeddings) {
    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto& layer = layers_[i];
      hidden_states = layer->forward(hidden_states);
    }
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  std::vector<SiglipEncoderLayer> layers_;
};
TORCH_MODULE(SiglipEncoder);

class SiglipVisionTransformerImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embeddings_ =
        register_module("embeddings", SiglipVisionEmbeddings(context));
    post_layer_norm_ = register_module(
        "post_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    post_layer_norm_->weight.set_data(post_layer_norm_->weight.to(options));
    post_layer_norm_->bias.set_data(post_layer_norm_->bias.to(options));
    encoder_ = register_module("encoder", SiglipEncoder(context));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    auto hidden_states = embeddings_->forward(pixel_values);
    auto encoder_output = encoder_->forward(hidden_states);
    auto last_hidden_state = post_layer_norm_->forward(encoder_output);
    return last_hidden_state;
  }

  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    weight::load_weight(state_dict,
                        "post_layernorm.weight",
                        post_layer_norm_->weight,
                        post_layer_norm_weight_loaded_);

    weight::load_weight(state_dict,
                        "post_layernorm.bias",
                        post_layer_norm_->bias,
                        post_layer_norm_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    CHECK(post_layer_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.weight";
    CHECK(post_layer_norm_bias_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.bias";
  }

 private:
  bool post_layer_norm_weight_loaded_ = false;
  bool post_layer_norm_bias_loaded_ = false;
  SiglipVisionEmbeddings embeddings_ = nullptr;
  SiglipEncoder encoder_ = nullptr;
  torch::nn::LayerNorm post_layer_norm_ = nullptr;
};
TORCH_MODULE(SiglipVisionTransformer);

class SiglipVisionModelImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    transformer_ =
        register_module("transformer", SiglipVisionTransformer(context));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    auto last_hidden_states = transformer_->forward(input_ids);
    return last_hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("vision_model."));
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      transformer_->load_state_dict(
          state_dict->get_dict_with_prefix("vision_model."));
    }

    transformer_->verify_loaded_weights("vision_model.");
  }

 private:
  SiglipVisionTransformer transformer_ = nullptr;
};
TORCH_MODULE(SiglipVisionModel);

REGISTER_MODEL_ARGS(SiglipVisionModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 1152);
  LOAD_ARG_OR(mm_image_size, "image_size", 384);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 27);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "num_channels", 3);
  LOAD_ARG_OR(mm_patch_size, "patch_size", 14);
  LOAD_ARG_OR(mm_hidden_act, "hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-6);
  LOAD_ARG_OR(mm_head_dim, "head_dim", 64);
});

}  // namespace xllm
