#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <regex>
#include <unordered_map>

#include "clip_text_model.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/siglip_encoder_layer.h"
#include "dit_linear.h"
#include "models/model_registry.h"
#include "processors/clip_image_processor.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm {

class CLIPVisionEmbeddingImpl : public torch::nn::Module {
 public:
  explicit CLIPVisionEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_dim_ = model_args.mm_hidden_size();
    image_size_ = model_args.mm_image_size();
    class_embedding_ = register_parameter("class_embedding",
                                          torch::randn({embed_dim_}, options));
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(model_args.mm_num_channels(),
                                                   embed_dim_,
                                                   model_args.mm_patch_size())
                              .stride(model_args.mm_patch_size())
                              .bias(false)));
    patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));

    auto num_patches =
        (model_args.mm_image_size() / model_args.mm_patch_size()) *
        (model_args.mm_image_size() / model_args.mm_patch_size());
    auto num_positions = num_patches + 1;
    position_embedding_ =
        register_parameter("position_embedding",
                           torch::randn({num_positions, embed_dim_}, options));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    int64_t batch_size = pixel_values.size(0);
    auto patch_embeds =
        patch_embedding_->forward(pixel_values).flatten(2).transpose(1, 2);
    auto class_embeds = class_embedding_.expand({batch_size, 1, -1});
    auto embeddings = torch::cat({class_embeds, patch_embeds}, 1);
    embeddings += position_embedding_.index({position_ids_});
    return embeddings;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto cls = state_dict.get_tensor("class_embedding");
    if (cls.defined()) {
      DCHECK_EQ(cls.sizes(), class_embedding_.sizes())
          << "class_embedding size mismatch for " << name();
      class_embedding_.data().copy_(cls);
      is_class_embedding_loaded = true;
    }

    const auto pos = state_dict.get_tensor("position_embedding.weight");
    if (pos.defined()) {
      CHECK_EQ(pos.sizes(), position_embedding_.sizes())
          << "position_embedding weight size mismatch for " << name();
      position_embedding_.data().copy_(pos);
      is_position_embedding_loaded = true;
    }

    const auto weight = state_dict.get_tensor("patch_embedding.weight");
    if (weight.defined()) {
      DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
          << "patch_embedding weight size mismatch for " << name();
      patch_embedding_->weight.data().copy_(weight);
      is_patch_embedding_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_class_embedding_loaded)
        << "weight is not loaded for " << prefix + "class_embedding";
    CHECK(is_position_embedding_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_patch_embedding_loaded)
        << "weight is not loaded for " << prefix + "patch_embedding.weight";
  }

 private:
  int64_t embed_dim_;
  int64_t image_size_;
  bool is_class_embedding_loaded{false};
  bool is_position_embedding_loaded{false};
  bool is_patch_embedding_loaded{false};

  torch::Tensor class_embedding_;
  torch::Tensor position_ids_;
  torch::nn::Conv2d patch_embedding_{nullptr};
  torch::Tensor position_embedding_{nullptr};
};
TORCH_MODULE(CLIPVisionEmbedding);

class CLIPVisionTransformerImpl : public torch::nn::Module {
 public:
  explicit CLIPVisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embeddings_ = register_module("embeddings", CLIPVisionEmbedding(context));
    pre_layernorm_ = register_module(
        "pre_layernorm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));

    encoder_ = register_module("encoder", CLIPEncoder(context));
    post_layernorm_ = register_module(
        "post_layernorm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
  }

  // std::vector<torch::Tensor> forward(const torch::Tensor& pixel_values) {
  torch::Tensor forward(const torch::Tensor& pixel_values) {
    auto hidden_states = embeddings_->forward(pixel_values);
    hidden_states = pre_layernorm_->forward(hidden_states);

    auto last_hidden_state = encoder_->forward(hidden_states, torch::Tensor());
    auto pool_output = last_hidden_state.select(1, 0);
    pool_output = post_layernorm_->forward(pool_output);
    // auto pool_output = post_layernorm_->forward(last_hidden_state);
    return pool_output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    weight::load_weight(state_dict,
                        "pre_layernorm.weight",
                        pre_layernorm_->weight,
                        pre_layernorm_weight_loaded_);
    weight::load_weight(state_dict,
                        "pre_layernorm.bias",
                        pre_layernorm_->bias,
                        pre_layernorm_bias_loaded_);
    weight::load_weight(state_dict,
                        "post_layernorm.weight",
                        post_layernorm_->weight,
                        post_layernorm_weight_loaded_);
    weight::load_weight(state_dict,
                        "post_layernorm.bias",
                        post_layernorm_->bias,
                        post_layernorm_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    CHECK(pre_layernorm_weight_loaded_)
        << "weight is not loaded for " << prefix + "pre_layernorm.weight";
    CHECK(pre_layernorm_bias_loaded_)
        << "weight is not loaded for " << prefix + "pre_layernorm.bias";
    CHECK(post_layernorm_weight_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.weight";
    CHECK(post_layernorm_bias_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.bias";
  }

 private:
  CLIPVisionEmbedding embeddings_{nullptr};
  torch::nn::LayerNorm pre_layernorm_{nullptr};
  CLIPEncoder encoder_{nullptr};
  torch::nn::LayerNorm post_layernorm_{nullptr};
  bool pre_layernorm_weight_loaded_ = false;
  bool pre_layernorm_bias_loaded_ = false;
  bool post_layernorm_weight_loaded_ = false;
  bool post_layernorm_bias_loaded_ = false;
};
TORCH_MODULE(CLIPVisionTransformer);

class CLIPVisionModelWithProjectionImpl : public torch::nn::Module {
 public:
  explicit CLIPVisionModelWithProjectionImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    transformer_ =
        register_module("transformer", CLIPVisionTransformer(context));
    projection_ = register_parameter(
        "projection",
        torch::randn(
            {model_args.mm_hidden_size(), model_args.mm_projection_dim()},
            options));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    auto last_hidden_states = transformer_->forward(input_ids);
    auto projected_output = torch::matmul(last_hidden_states, projection_);
    return projected_output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("vision_model."));
    const auto proj_weight = state_dict.get_tensor("projection");
    if (proj_weight.defined()) {
      DCHECK_EQ(projection_.sizes(), proj_weight.sizes())
          << "projection size mismatch for " << name();
      projection_.data().copy_(proj_weight);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + "vision_model.");
    DCHECK(projection_.defined())
        << "projection parameter not loaded for " << prefix + "projection";
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      transformer_->load_state_dict(
          state_dict->get_dict_with_prefix("vision_model."));
    }

    // verify
    transformer_->verify_loaded_weights("vision_model.");
    LOG(INFO) << "clip_vision_model loaded successfully.";
  }

 private:
  CLIPVisionTransformer transformer_{nullptr};
  torch::Tensor projection_;
};
TORCH_MODULE(CLIPVisionModelWithProjection);

REGISTER_MODEL_ARGS(CLIPVisionModelWithProjection, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "float32");
  LOAD_ARG_OR(model_type, "model_type", "clip_vision_model");
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(mm_projection_dim, "projection_dim", 1024);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(mm_hidden_act, "hidden_act", "gelu");
  LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(mm_image_size, "image_size", 224);
  LOAD_ARG_OR(mm_patch_size, "patch_size", 14);
  LOAD_ARG_OR(mm_num_channels, "num_channels", 3);
});
}  // namespace xllm
