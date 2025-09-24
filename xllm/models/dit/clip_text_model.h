
#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <regex>
#include <unordered_map>

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
#include "processors/qwen2_vl_image_processor.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm::hf {
// clip_text_model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip
torch::Tensor quick_gelu(torch::Tensor x) {
  return x * torch::sigmoid(1.702 * x);
}

// causal_mask (batch_size, 1, seq_len, seq_len)
torch::Tensor _create_4d_causal_attention_mask(torch::IntArrayRef input_shape,
                                               torch::Dtype dtype,
                                               torch::Device device) {
  const int64_t bsz = input_shape[0];
  const int64_t tgt_len = input_shape[1];

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  auto causal_mask = torch::full(
      {tgt_len, tgt_len}, -std::numeric_limits<double>::infinity(), options);
  causal_mask.triu_(1);
  causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);
  causal_mask = causal_mask.expand({bsz, 1, tgt_len, tgt_len});
  return causal_mask;
}

class CLIPVLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  CLIPVLInputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
  }
  void process(std::string& prompt, const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();
    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();
    if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;
    auto merge_length = merge_size_ * merge_size_;
    int total_image_token = 0;
    if (image_grid_thw.defined()) {
      auto count = image_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
    }
    int total_video_token = 0;
    if (video_grid_thw.defined()) {
      auto count = video_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
    }
    size_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);
    int image_index = 0;
    int video_index = 0;
    const torch::Tensor* grid_thw = nullptr;
    const std::string* token = nullptr;
    int* index = 0;
    size_t begin = 0;
    auto pair = _find_vision_token(prompt, begin);
    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);
      if (pair.first == TokenType::IMAGE) {
        grid_thw = &image_grid_thw;
        token = &image_token_;
        index = &image_index;
      } else if (pair.first == TokenType::VIDEO) {
        grid_thw = &video_grid_thw;
        token = &video_token_;
        index = &video_index;
      } else {
        assert(false);
      }
      auto token_num = (*grid_thw)[(*index)].prod().item<int>() / merge_length;
      while (token_num--) data.append(*token);
      ++(*index);
      begin = pair.second + token->size();
      pair = _find_vision_token(prompt, begin);
    }
    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
    prompt = std::move(data);
  }

 private:
  std::pair<TokenType, size_t> _find_vision_token(const std::string& prompt,
                                                  size_t begin) {
    auto img_pos = prompt.find(image_token_, begin);
    auto vid_pos = prompt.find(video_token_, begin);
    if (img_pos == std::string::npos && vid_pos == std::string::npos)
      return {TokenType::INVALID, std::string::npos};
    else if (vid_pos == std::string::npos)
      return {TokenType::IMAGE, img_pos};
    else if (img_pos == std::string::npos)
      return {TokenType::VIDEO, vid_pos};
    else
      return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                               : std::make_pair(TokenType::VIDEO, vid_pos);
  }

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  int merge_size_ = 0;
};

class CLIPTextEmbeddingImpl : public torch::nn::Module {
 public:
  CLIPTextEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    token_embedding_ = register_module(
        "token_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(
            model_args.clip_vocab_size(), model_args.clip_hidden_size())));
    token_embedding_->weight.set_data(token_embedding_->weight.to(options));
    position_embedding_ = register_parameter(
        "position_embedding",
        torch::randn({model_args.clip_max_position_embeddings(),
                      model_args.clip_hidden_size()},
                     options));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(
            0, model_args.clip_max_position_embeddings(), torch::kLong)
            .unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    int64_t batch_size = input_ids.size(0);
    int64_t seq_length = input_ids.size(1);
    int64_t max_position_embedding = position_embedding_.size(0);
    CHECK(seq_length <= max_position_embedding);

    torch::Tensor inputs_embeds = token_embedding_->forward(input_ids);
    torch::Tensor position_ids = position_ids_.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, seq_length)});
    torch::Tensor embeddings =
        inputs_embeds + position_embedding_.index({position_ids});
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto tok_emb = state_dict.get_tensor("token_embedding.weight");
    if (tok_emb.defined()) {
      DCHECK_EQ(token_embedding_->weight.sizes(), tok_emb.sizes())
          << "patch_embedding weight size mismatch for " << name();
      token_embedding_->weight.data().copy_(tok_emb);
      is_token_embedding_loaded = true;
    }

    auto pos = state_dict.get_tensor("position_embedding.weight");
    if (pos.defined()) {
      CHECK_EQ(pos.sizes(), position_embedding_.sizes())
          << "position_embedding weight size mismatch for " << name();
      position_embedding_.data().copy_(pos);
      is_position_embedding_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_position_embedding_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_token_embedding_loaded)
        << "weight is not loaded for " << prefix + "token_embedding.weight";
  }

 private:
  bool is_position_embedding_loaded{false};
  bool is_token_embedding_loaded{false};
  torch::Tensor position_ids_;
  torch::nn::Embedding token_embedding_{nullptr};
  torch::Tensor position_embedding_{nullptr};
};
TORCH_MODULE(CLIPTextEmbedding);

class CLIPMLPImpl : public torch::nn::Module {
 public:
  CLIPMLPImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    act_ = register_module("act", torch::nn::Functional(quick_gelu));

    fc1_ = register_module("fc1",
                           DiTLinear(model_args.clip_hidden_size(),
                                     model_args.clip_intermediate_size(),
                                     true));
    fc2_ = register_module("fc2",
                           DiTLinear(model_args.clip_intermediate_size(),
                                     model_args.clip_hidden_size(),
                                     true));

    fc1_->weight.set_data(fc1_->weight.to(options));
    fc2_->weight.set_data(fc2_->weight.to(options));
    fc1_->bias.set_data(fc1_->bias.to(options));
    fc2_->bias.set_data(fc2_->bias.to(options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto fc1_weight = state_dict.get_tensor("fc1.weight");
    if (fc1_weight.defined()) {
      DCHECK_EQ(fc1_weight.sizes(), fc1_->weight.sizes())
          << "fc1 weight size mismatch";
      fc1_->weight.data().copy_(fc1_weight);
      fc1_weight_loaded_ = true;
    }

    const auto fc1_bias = state_dict.get_tensor("fc1.bias");
    if (fc1_bias.defined()) {
      DCHECK_EQ(fc1_bias.sizes(), fc1_->bias.sizes())
          << "fc1 bias size mismatch";
      fc1_->bias.data().copy_(fc1_bias);
      fc1_bias_loaded_ = true;
    }

    const auto fc2_weight = state_dict.get_tensor("fc2.weight");
    if (fc2_weight.defined()) {
      DCHECK_EQ(fc2_weight.sizes(), fc2_->weight.sizes())
          << "fc2 weight size mismatch";
      fc2_->weight.data().copy_(fc2_weight);
      fc2_weight_loaded_ = true;
    }

    const auto fc2_bias = state_dict.get_tensor("fc2.bias");
    if (fc2_bias.defined()) {
      DCHECK_EQ(fc2_bias.sizes(), fc2_->bias.sizes())
          << "fc2 bias size mismatch";
      fc2_->bias.data().copy_(fc2_bias);
      fc2_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(fc1_weight_loaded_)
        << "weight is not loaded for " << prefix + "fc1.weight";
    CHECK(fc1_bias_loaded_)
        << "weight is not loaded for " << prefix + "fc1.bias";
    CHECK(fc2_weight_loaded_)
        << "weight is not loaded for " << prefix + "fc2.weight";
    CHECK(fc2_bias_loaded_)
        << "weight is not loaded for " << prefix + "fc2.bias";
  }

 private:
  torch::nn::Functional act_{nullptr};
  DiTLinear fc1_{nullptr};
  DiTLinear fc2_{nullptr};
  bool fc1_weight_loaded_{false};
  bool fc1_bias_loaded_{false};
  bool fc2_weight_loaded_{false};
  bool fc2_bias_loaded_{false};
};
TORCH_MODULE(CLIPMLP);

// TODO: Optimize CLIPAttention
class CLIPAttentionImpl : public torch::nn::Module {
 public:
  CLIPAttentionImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    CHECK(model_args.clip_hidden_size() %
              model_args.clip_num_attention_heads() ==
          0);
    head_dim_ = model_args.clip_head_dim();
    embed_dim_ = model_args.clip_hidden_size();
    num_heads_ = model_args.clip_num_attention_heads();
    const int64_t n_local_heads = num_heads_;

    qkv_sizes_ = {n_local_heads * model_args.clip_head_dim(),
                  n_local_heads * model_args.clip_head_dim(),
                  n_local_heads * model_args.clip_head_dim()};

    scale_ = 1.0f / std::sqrt(static_cast<float>(model_args.clip_head_dim()));
    q_proj_ = register_module(
        "q_proj",
        DiTLinear(model_args.clip_hidden_size(), num_heads_ * head_dim_, true));
    k_proj_ = register_module(
        "k_proj",
        DiTLinear(model_args.clip_hidden_size(), num_heads_ * head_dim_, true));
    v_proj_ = register_module(
        "v_proj",
        DiTLinear(model_args.clip_hidden_size(), num_heads_ * head_dim_, true));
    o_proj_ = register_module("o_proj",
                              DiTLinear(model_args.clip_hidden_size(),
                                        model_args.clip_hidden_size(),
                                        true));

    q_proj_->weight.set_data(q_proj_->weight.to(options));
    k_proj_->weight.set_data(k_proj_->weight.to(options));
    v_proj_->weight.set_data(v_proj_->weight.to(options));
    o_proj_->weight.set_data(o_proj_->weight.to(options));

    q_proj_->bias.set_data(q_proj_->bias.to(options));
    k_proj_->bias.set_data(k_proj_->bias.to(options));
    v_proj_->bias.set_data(v_proj_->bias.to(options));
    o_proj_->bias.set_data(o_proj_->bias.to(options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        torch::Tensor causal_mask) {
    auto bsz = hidden_states.size(0);
    auto tgt_len = hidden_states.size(1);

    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // [batch_size, num_heads, seq_len, head_dim]
    query_states = shape(query_states, tgt_len, bsz);
    key_states = shape(key_states, -1, bsz);
    value_states = shape(value_states, -1, bsz);

    auto src_len = key_states.size(1);
    auto attn_weights =
        torch::matmul(query_states, key_states.transpose(-1, -2)) * scale_;
    if (causal_mask.defined()) attn_weights = attn_weights + causal_mask;
    attn_weights = torch::softmax(attn_weights, -1);
    auto attn_output = torch::matmul(attn_weights, value_states);

    DCHECK_EQ(attn_output.sizes(),
              torch::IntArrayRef({bsz * num_heads_, tgt_len, head_dim_}));
    attn_output =
        attn_output
            .view(torch::IntArrayRef({bsz, num_heads_, tgt_len, head_dim_}))
            .transpose(1, 2)
            .contiguous();
    attn_output =
        attn_output.view(torch::IntArrayRef({bsz, tgt_len, embed_dim_}));

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto q_proj_weight = state_dict.get_tensor("q_proj.weight");
    if (q_proj_weight.defined()) {
      DCHECK_EQ(q_proj_weight.sizes(), q_proj_->weight.sizes())
          << "q_proj weight size mismatch";
      q_proj_->weight.data().copy_(q_proj_weight);
      q_proj_weight_loaded_ = true;
    }

    const auto q_proj_bias = state_dict.get_tensor("q_proj.bias");
    if (q_proj_bias.defined()) {
      DCHECK_EQ(q_proj_bias.sizes(), q_proj_->bias.sizes())
          << "q_proj bias size mismatch";
      q_proj_->bias.data().copy_(q_proj_bias);
      q_proj_bias_loaded_ = true;
    }

    const auto k_proj_weight = state_dict.get_tensor("k_proj.weight");
    if (k_proj_weight.defined()) {
      DCHECK_EQ(k_proj_weight.sizes(), k_proj_->weight.sizes())
          << "k_proj weight size mismatch";
      k_proj_->weight.data().copy_(k_proj_weight);
      k_proj_weight_loaded_ = true;
    }

    const auto k_proj_bias = state_dict.get_tensor("k_proj.bias");
    if (k_proj_bias.defined()) {
      DCHECK_EQ(k_proj_bias.sizes(), k_proj_->bias.sizes())
          << "k_proj bias size mismatch";
      k_proj_->bias.data().copy_(k_proj_bias);
      k_proj_bias_loaded_ = true;
    }

    const auto v_proj_weight = state_dict.get_tensor("v_proj.weight");
    if (v_proj_weight.defined()) {
      DCHECK_EQ(v_proj_weight.sizes(), v_proj_->weight.sizes())
          << "v_proj weight size mismatch";
      v_proj_->weight.data().copy_(v_proj_weight);
      v_proj_weight_loaded_ = true;
    }

    const auto v_proj_bias = state_dict.get_tensor("v_proj.bias");
    if (v_proj_bias.defined()) {
      DCHECK_EQ(v_proj_bias.sizes(), v_proj_->bias.sizes())
          << "v_proj bias size mismatch";
      v_proj_->bias.data().copy_(v_proj_bias);
      v_proj_bias_loaded_ = true;
    }

    const auto o_proj_weight = state_dict.get_tensor("out_proj.weight");
    if (o_proj_weight.defined()) {
      DCHECK_EQ(o_proj_weight.sizes(), o_proj_->weight.sizes())
          << "o_proj weight size mismatch";
      o_proj_->weight.data().copy_(o_proj_weight);
      o_proj_weight_loaded_ = true;
    }

    const auto o_proj_bias = state_dict.get_tensor("out_proj.bias");
    if (o_proj_bias.defined()) {
      DCHECK_EQ(o_proj_bias.sizes(), o_proj_->bias.sizes())
          << "o_proj bias size mismatch";
      o_proj_->bias.data().copy_(o_proj_bias);
      o_proj_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(q_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "q_proj.weight";
    CHECK(q_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "q_proj.bias";
    CHECK(k_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "k_proj.weight";
    CHECK(k_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "k_proj.bias";
    CHECK(v_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "v_proj.weight";
    CHECK(v_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "v_proj.bias";
    CHECK(o_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "out_proj.weight";
    CHECK(o_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "out_proj.bias";
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
  std::vector<int64_t> qkv_sizes_;

  DiTLinear o_proj_{nullptr};
  DiTLinear q_proj_{nullptr};
  DiTLinear k_proj_{nullptr};
  DiTLinear v_proj_{nullptr};

  bool q_proj_weight_loaded_{false};
  bool q_proj_bias_loaded_{false};
  bool k_proj_weight_loaded_{false};
  bool k_proj_bias_loaded_{false};
  bool v_proj_weight_loaded_{false};
  bool v_proj_bias_loaded_{false};
  bool o_proj_weight_loaded_{false};
  bool o_proj_bias_loaded_{false};
};
TORCH_MODULE(CLIPAttention);

class CLIPEncoderLayerImpl : public torch::nn::Module {
 public:
  CLIPEncoderLayerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attn_ = register_module("self_attn", CLIPAttention(context));
    layer_norm1_ = register_module(
        "layer_norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.clip_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.clip_layer_norm_eps())));
    layer_norm2_ = register_module(
        "layer_norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.clip_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.clip_layer_norm_eps())));
    layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
    layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
    layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
    layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
    mlp_ = register_module("mlp", CLIPMLP(context));
  }

  // TODO: self_attn, attention_mask
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        torch::Tensor causal_mask) {
    auto residual = hidden_states;
    const auto& layer_norm1 = layer_norm1_(hidden_states);
    auto h = self_attn_(layer_norm1, causal_mask) + residual;
    residual = h;
    h = layer_norm2_(h);
    h = mlp_(h);
    h += residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));

    const auto& layer_norm1_weight =
        state_dict.get_tensor("layer_norm1.weight");
    if (layer_norm1_weight.defined()) {
      DCHECK_EQ(layer_norm1_weight.sizes(), layer_norm1_->weight.sizes())
          << "layer_norm1 weight size mismatch";
      layer_norm1_->weight.data().copy_(layer_norm1_weight);
      layer_norm1_weight_loaded_ = true;
    }

    const auto layer_norm1_bias = state_dict.get_tensor("layer_norm1.bias");
    if (layer_norm1_bias.defined()) {
      DCHECK_EQ(layer_norm1_bias.sizes(), layer_norm1_->bias.sizes())
          << "layer_norm1 bias size mismatch";
      layer_norm1_->bias.data().copy_(layer_norm1_bias);
      layer_norm1_bias_loaded_ = true;
    }

    const auto layer_norm2_weight = state_dict.get_tensor("layer_norm2.weight");
    if (layer_norm2_weight.defined()) {
      DCHECK_EQ(layer_norm2_weight.sizes(), layer_norm2_->weight.sizes())
          << "layer_norm2 weight size mismatch";
      layer_norm2_->weight.data().copy_(layer_norm2_weight);
      layer_norm2_weight_loaded_ = true;
    }

    const auto layer_norm2_bias = state_dict.get_tensor("layer_norm2.bias");
    if (layer_norm2_bias.defined()) {
      DCHECK_EQ(layer_norm2_bias.sizes(), layer_norm2_->bias.sizes())
          << "layer_norm2 bias size mismatch";
      layer_norm2_->bias.data().copy_(layer_norm2_bias);
      layer_norm2_bias_loaded_ = true;
    }

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
  bool layer_norm1_weight_loaded_{false};
  bool layer_norm1_bias_loaded_{false};
  bool layer_norm2_weight_loaded_{false};
  bool layer_norm2_bias_loaded_{false};

  torch::nn::LayerNorm layer_norm1_{nullptr};
  torch::nn::LayerNorm layer_norm2_{nullptr};
  CLIPAttention self_attn_{nullptr};
  CLIPMLP mlp_{nullptr};
};
TORCH_MODULE(CLIPEncoderLayer);

class CLIPEncoderImpl : public torch::nn::Module {
 public:
  CLIPEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.clip_num_hidden_layers());
    for (int32_t i = 0; i < model_args.clip_num_hidden_layers(); i++) {
      auto block = CLIPEncoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // Output hidden states for last intermediate layers
  torch::Tensor forward(const torch::Tensor& embeddings,
                        torch::Tensor causal_mask) {
    bool output_hidden_states = false;
    bool output_attentions = false;
    c10::optional<torch::Tensor> attention_mask = c10::nullopt;
    c10::optional<torch::Tensor> head_mask = c10::nullopt;
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    std::vector<torch::Tensor> encoder_states;

    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      encoder_states.emplace_back(hidden_states);
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states, causal_mask);
    }
    if (output_hidden_states) encoder_states.emplace_back(hidden_states);

    std::vector<torch::Tensor> outputs = {hidden_states};
    if (output_hidden_states) {
      // todo
    }
    if (output_attentions) {
      // todo
    }
    return outputs[0];
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
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<CLIPEncoderLayer> layers_;
};
TORCH_MODULE(CLIPEncoder);

class CLIPTextTransformerImpl : public torch::nn::Module {
 public:
  CLIPTextTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embeddings_ = register_module("embeddings", CLIPTextEmbedding(context));
    final_layer_norm_ = register_module(
        "final_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.clip_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.clip_layer_norm_eps())));
    final_layer_norm_->weight.set_data(final_layer_norm_->weight.to(options));
    final_layer_norm_->bias.set_data(final_layer_norm_->bias.to(options));
    encoder_ = register_module("encoder", CLIPEncoder(context));
    eos_token_id = model_args.clip_eos_token_id();
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    if (!input_ids.defined()) {
      throw std::runtime_error("You have to specify input_ids");
    }
    auto input_shape = input_ids.sizes();
    auto reshaped_input_ids = input_ids.view({-1, input_shape.back()});
    auto hidden_states = embeddings_->forward(reshaped_input_ids);
    auto causal_mask = _create_4d_causal_attention_mask(
        {input_shape[0], input_shape[1]},
        torch::typeMetaToScalarType(hidden_states.dtype()),
        hidden_states.device());
    auto encoder_output = encoder_->forward(hidden_states, causal_mask);
    auto last_hidden_state = final_layer_norm_->forward(encoder_output);
    return last_hidden_state;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));

    const auto final_layer_norm_weight =
        state_dict.get_tensor("final_layer_norm.weight");
    if (final_layer_norm_weight.defined()) {
      DCHECK_EQ(final_layer_norm_weight.sizes(),
                final_layer_norm_->weight.sizes())
          << "final_layer_norm weight size mismatch";
      final_layer_norm_->weight.data().copy_(final_layer_norm_weight);
      final_layer_norm_weight_loaded_ = true;
    }

    const auto final_layer_norm_bias =
        state_dict.get_tensor("final_layer_norm.bias");
    if (final_layer_norm_bias.defined()) {
      DCHECK_EQ(final_layer_norm_bias.sizes(), final_layer_norm_->bias.sizes())
          << "final_layer_norm bias size mismatch";
      final_layer_norm_->bias.data().copy_(final_layer_norm_bias);
      final_layer_norm_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    CHECK(final_layer_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "final_layer_norm.weight";
    CHECK(final_layer_norm_bias_loaded_)
        << "weight is not loaded for " << prefix + "final_layer_norm.bias";
  }

 private:
  int64_t eos_token_id;
  bool final_layer_norm_weight_loaded_{false};
  bool final_layer_norm_bias_loaded_{false};
  CLIPTextEmbedding embeddings_{nullptr};
  CLIPEncoder encoder_{nullptr};
  torch::nn::LayerNorm final_layer_norm_{nullptr};
};
TORCH_MODULE(CLIPTextTransformer);

class CLIPTextModelImpl : public torch::nn::Module {
 public:
  CLIPTextModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    eos_token_id = model_args.clip_eos_token_id();
    transformer_ = register_module("transformer", CLIPTextTransformer(context));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    auto last_hidden_states = transformer_->forward(input_ids);
    int64_t batch_size = last_hidden_states.size(0);
    auto device = last_hidden_states.device();
    torch::Tensor batch_indices = torch::arange(batch_size, device);
    torch::Tensor end_pos;
    if (eos_token_id == 2) {
      auto argmax_result = input_ids.to(device).max(1);
      end_pos = std::get<1>(argmax_result);
    } else {
      torch::Tensor eos_mask =
          (input_ids == eos_token_id).to(device, torch::kInt);
      auto argmax_result = eos_mask.max(1);
      end_pos = std::get<1>(argmax_result);
    }
    torch::Tensor pooled_output =
        last_hidden_states.index({batch_indices, end_pos});
    return pooled_output;
  }

  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("text_model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + ".");
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    LOG(INFO) << "Loading CLIPTextModel from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      transformer_->load_state_dict(
          state_dict->get_dict_with_prefix("text_model."));
    }

    // verify
    transformer_->verify_loaded_weights("text_model.");
    LOG(INFO) << "clip_text_model loaded successfully.";
  }

 private:
  int64_t eos_token_id;
  CLIPTextTransformer transformer_{nullptr};
};
TORCH_MODULE(CLIPTextModel);

REGISTER_MODEL_ARGS(CLIPTextModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(clip_vocab_size, "vocab_size", 49408);
  LOAD_ARG_OR(clip_hidden_size, "hidden_size", 768);
  LOAD_ARG_OR(clip_intermediate_size, "intermediate_size", 3072);
  LOAD_ARG_OR(clip_projection_dim, "projection_dim", 768);
  LOAD_ARG_OR(clip_num_hidden_layers, "num_hidden_layers", 12);
  LOAD_ARG_OR(clip_num_attention_heads, "num_attention_heads", 12);
  LOAD_ARG_OR(clip_max_position_embeddings, "max_position_embeddings", 77);
  LOAD_ARG_OR(clip_hidden_act, "hidden_act", "quick_gelu");
  LOAD_ARG_OR(clip_layer_norm_eps, "layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(clip_initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(clip_initializer_factor, "initializer_factor", 1.0f);
  LOAD_ARG_OR(clip_pad_token_id, "pad_token_id", 1);
  LOAD_ARG_OR(clip_bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(clip_eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(clip_head_dim, "head_dim", 64);
});
}  // namespace xllm
