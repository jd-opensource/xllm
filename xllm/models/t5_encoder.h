#pragma once
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "framework/context.h"
#include "model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
namespace xllm::hf {
// T5 model compatible with huggingface weights
//    ref to:
//   https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5
class T5LayerNormImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  double variance_epsilon;
  torch::Device device_;
  torch::ScalarType dtype_;

 public:
  T5LayerNormImpl(int64_t hidden_size,
                  double eps = 1e-6,
                  torch::Device device = torch::kCPU,
                  torch::ScalarType dtype = torch::kBFloat16)
      : variance_epsilon(eps), device_(device), dtype_(dtype) {
    weight = register_parameter(
        "weight", torch::ones({hidden_size}).to(device_).to(dtype_));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    auto variance = hidden_states.to(dtype_).pow(2).mean(-1, true);
    hidden_states = hidden_states * torch::rsqrt(variance + variance_epsilon);
    if (weight.dtype() == torch::kFloat16 ||
        weight.dtype() == torch::kBFloat16) {
      hidden_states = hidden_states.to(weight.dtype());
    }
    return weight * hidden_states;
  }
  void load_state_dict(const StateDict& state_dict) {
    auto weight_tensor = state_dict.get_tensor("weight");
    if (weight_tensor.defined()) {
      DCHECK_EQ(weight.sizes(), weight_tensor.sizes())
          << "weight size mismatch: expected " << weight.sizes() << " but got "
          << weight_tensor.sizes();
      weight.data().copy_(weight_tensor);
    }
  }
};
TORCH_MODULE(T5LayerNorm);
torch::Tensor gelu_new(const torch::Tensor& x) {
  // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
  return 0.5 * x *
         (1.0 +
          torch::tanh(sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3))));
}
class T5DenseInterface : public torch::nn::Module {
 public:
  virtual torch::Tensor forward(const torch::Tensor& hidden_states) = 0;
  virtual std::shared_ptr<T5DenseInterface> clone(
      const torch::Device& device = torch::kCPU) const = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;
};
class T5DenseActDenseImpl : public T5DenseInterface {
 private:
  torch::nn::Linear wi_{nullptr};
  torch::nn::Linear wo_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  torch::nn::Functional act_{nullptr};
  torch::Device device_;
  torch::ScalarType dtype_;

 public:
  T5DenseActDenseImpl(int64_t d_model,
                      int64_t d_ff,
                      double dropout_rate,
                      const std::string& dense_act_fn,
                      torch::Device device = torch::kCPU,
                      torch::ScalarType dtype = torch::kBFloat16)
      : device_(device), dtype_(dtype) {
    wi_ = register_module(
        "wi",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ff).bias(false)));
    wo_ = register_module(
        "wo",
        torch::nn::Linear(torch::nn::LinearOptions(d_ff, d_model).bias(false)));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout_rate));
    if (dense_act_fn == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (dense_act_fn == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      throw std::invalid_argument("Unsupported activation function: " +
                                  dense_act_fn);
    }
  }
  void load_state_dict(const StateDict& state_dict) {
    // wi
    const auto wi_weight = state_dict.get_tensor("wi.weight");
    if (wi_weight.defined()) {
      DCHECK_EQ(wi_weight.sizes(), wi_->weight.sizes())
          << "wi weight size mismatch";
      wi_->weight.data().copy_(wi_weight);
    }

    // wo
    const auto wo_weight = state_dict.get_tensor("wo.weight");
    if (wo_weight.defined()) {
      DCHECK_EQ(wo_weight.sizes(), wo_->weight.sizes())
          << "wo weight size mismatch";
      wo_->weight.data().copy_(wo_weight);
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden = wi_->forward(hidden_states);
    hidden = act_(hidden);
    hidden = dropout_->forward(hidden);
    if (wo_->weight.dtype() != torch::kInt8 &&
        hidden.dtype() != wo_->weight.dtype()) {
      hidden = hidden.to(wo_->weight.dtype());
    }
    hidden = wo_->forward(hidden);
    return hidden;
  }
  std::shared_ptr<T5DenseInterface> clone(
      const torch::Device& device = torch::kCPU) const override {
    auto clone = std::make_shared<T5DenseActDenseImpl>(*this);
    clone->to(device);
    return clone;
  }
};
class T5DenseGatedActDenseImpl : public T5DenseInterface {
 private:
  torch::nn::Linear wi_0_{nullptr};
  torch::nn::Linear wi_1_{nullptr};
  torch::nn::Linear wo_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  torch::nn::Functional act_{nullptr};
  torch::Device device_;
  torch::ScalarType dtype_;

 public:
  T5DenseGatedActDenseImpl(int64_t d_model,
                           int64_t d_ff,
                           double dropout_rate,
                           const std::string& dense_act_fn,
                           torch::Device device = torch::kCPU,
                           torch::ScalarType dtype = torch::kBFloat16)
      : device_(device), dtype_(dtype) {
    wi_0_ = register_module(
        "wi_0",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ff).bias(false)));
    wi_1_ = register_module(
        "wi_1",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ff).bias(false)));
    wo_ = register_module(
        "wo",
        torch::nn::Linear(torch::nn::LinearOptions(d_ff, d_model).bias(false)));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout_rate));
    if (dense_act_fn == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (dense_act_fn == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      throw std::invalid_argument("Unsupported activation function: " +
                                  dense_act_fn);
    }
  }
  void load_state_dict(const StateDict& state_dict) {
    // wi_0
    const auto wi_0_weight = state_dict.get_tensor("wi_0.weight");
    if (wi_0_weight.defined()) {
      DCHECK_EQ(wi_0_weight.sizes(), wi_0_->weight.sizes())
          << "wi_0 weight size mismatch";
      wi_0_->weight.data().copy_(wi_0_weight);
    }
    // wi_1
    const auto wi_1_weight = state_dict.get_tensor("wi_1.weight");
    if (wi_1_weight.defined()) {
      DCHECK_EQ(wi_1_weight.sizes(), wi_1_->weight.sizes())
          << "wi_1 weight size mismatch";
      wi_1_->weight.data().copy_(wi_1_weight);
    }
    // wo
    const auto wo_weight = state_dict.get_tensor("wo.weight");
    if (wo_weight.defined()) {
      DCHECK_EQ(wo_weight.sizes(), wo_->weight.sizes())
          << "wo weight size mismatch";
      wo_->weight.data().copy_(wo_weight);
    }
  }
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden_gelu = act_(wi_0_->forward(hidden_states));
    torch::Tensor hidden_linear = wi_1_->forward(hidden_states);
    torch::Tensor new_hidden_states = hidden_gelu * hidden_linear;
    new_hidden_states = dropout_->forward(new_hidden_states);
    if (wo_->weight.dtype() != torch::kInt8 &&
        new_hidden_states.dtype() != wo_->weight.dtype()) {
      new_hidden_states = new_hidden_states.to(wo_->weight.dtype());
    }
    new_hidden_states = wo_->forward(new_hidden_states);
    return new_hidden_states;
  }

  std::shared_ptr<T5DenseInterface> clone(
      const torch::Device& device = torch::kCPU) const override {
    auto clone = std::make_shared<T5DenseGatedActDenseImpl>(*this);
    clone->to(device);
    return clone;
  }
};
class T5LayerFFNImpl : public torch::nn::Module {
 private:
  std::shared_ptr<T5DenseInterface> dense_relu_dense_{nullptr};
  T5LayerNorm layer_norm_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  torch::Device device_;
  torch::ScalarType dtype_;

 public:
  T5LayerFFNImpl(int64_t d_model,
                 int64_t d_ff,
                 double dropout_rate,
                 const std::string& dense_act_fn,
                 bool is_gated_act,
                 double layer_norm_epsilon = 1e-6,
                 torch::Device device = torch::kCPU,
                 torch::ScalarType dtype = torch::kBFloat16)
      : layer_norm_(d_model, layer_norm_epsilon, device, dtype),
        device_(device),
        dtype_(dtype) {
    if (is_gated_act) {
      dense_relu_dense_ = register_module(
          "DenseReluDense",
          std::make_shared<T5DenseGatedActDenseImpl>(
              d_model, d_ff, dropout_rate, dense_act_fn, device, dtype));
    } else {
      dense_relu_dense_ = register_module(
          "DenseReluDense",
          std::make_shared<T5DenseActDenseImpl>(
              d_model, d_ff, dropout_rate, dense_act_fn, device, dtype));
    }
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout_rate));
  }
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor forwarded_states = layer_norm_->forward(hidden_states);
    forwarded_states = dense_relu_dense_->forward(forwarded_states);
    torch::Tensor output = hidden_states + dropout_->forward(forwarded_states);
    return output;
  }
  void load_state_dict(const StateDict& state_dict) {
    dense_relu_dense_->load_state_dict(
        state_dict.get_dict_with_prefix("DenseReluDense."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }
};
TORCH_MODULE(T5LayerFFN);
std::pair<std::unordered_set<int64_t>, torch::Tensor>
find_pruneable_heads_and_indices(
    const std::vector<int64_t>& heads,
    int64_t n_heads,
    int64_t head_size,
    const std::unordered_set<int64_t>& already_pruned_heads,
    torch::ScalarType dtype = torch::kBFloat16) {
  std::unordered_set<int64_t> heads_to_prune;
  for (int64_t h : heads) {
    if (already_pruned_heads.find(h) == already_pruned_heads.end()) {
      heads_to_prune.insert(h);
    }
  }
  torch::Tensor mask = torch::ones({n_heads, head_size}, dtype);

  for (int64_t head : heads_to_prune) {
    int64_t adjusted_head = head;
    for (int64_t pruned : already_pruned_heads) {
      if (pruned < head) {
        adjusted_head--;
      }
    }
    mask[adjusted_head] = 0.0f;
  }
  mask = mask.view(-1).contiguous().eq(1.0f);
  torch::Tensor index = torch::arange(mask.numel(), torch::kLong).index({mask});

  return {heads_to_prune, index};
}
torch::nn::Linear prune_linear_layer(const torch::nn::Linear& layer,
                                     const torch::Tensor& index,
                                     int64_t dim = 0) {
  torch::Device device = layer->weight.device();
  torch::Tensor pruned_weight =
      layer->weight.index_select(dim, index.to(device)).detach().clone();
  c10::optional<torch::Tensor> pruned_bias = c10::nullopt;
  if (layer->bias.defined()) {
    if (dim == 1) {
      pruned_bias = layer->bias.detach().clone();
    } else {
      pruned_bias = layer->bias.index({index.to(device)}).detach().clone();
    }
  }
  torch::nn::LinearOptions options(pruned_weight.size(1),  // in_features
                                   pruned_weight.size(0)   // out_features
  );
  options.bias(pruned_bias.has_value());

  torch::nn::Linear new_layer(options);
  new_layer->to(device);
  new_layer->weight.requires_grad_(false);
  new_layer->weight.copy_(pruned_weight.contiguous());
  new_layer->weight.requires_grad_(true);

  if (pruned_bias.has_value()) {
    new_layer->bias.requires_grad_(false);
    new_layer->bias.copy_(pruned_bias.value().contiguous());
    new_layer->bias.requires_grad_(true);
  }

  return new_layer;
}
class T5AttentionImpl : public torch::nn::Module {
 public:
  T5AttentionImpl(int64_t d_model,
                  int64_t d_kv,
                  int64_t num_heads,
                  double dropout_rate,
                  bool has_relative_attention_bias,
                  int64_t relative_attention_num_buckets,
                  int64_t relative_attention_max_distance,
                  c10::optional<int64_t> layer_idx = c10::nullopt,
                  torch::Device device = torch::kCPU,
                  torch::ScalarType dtype = torch::kBFloat16)
      : has_relative_attention_bias_(has_relative_attention_bias),
        relative_attention_num_buckets_(relative_attention_num_buckets),
        relative_attention_max_distance_(relative_attention_max_distance),
        d_model_(d_model),
        key_value_proj_dim_(d_kv),
        n_heads_(num_heads),
        dropout_(dropout_rate),
        inner_dim_(num_heads * d_kv),
        layer_idx_(layer_idx),
        device_(device),
        dtype_(dtype) {
    q_ = register_module(
        "q",
        torch::nn::Linear(
            torch::nn::LinearOptions(d_model_, inner_dim_).bias(false)));
    k_ = register_module(
        "k",
        torch::nn::Linear(
            torch::nn::LinearOptions(d_model_, inner_dim_).bias(false)));
    v_ = register_module(
        "v",
        torch::nn::Linear(
            torch::nn::LinearOptions(d_model_, inner_dim_).bias(false)));
    o_ = register_module(
        "o",
        torch::nn::Linear(
            torch::nn::LinearOptions(inner_dim_, d_model_).bias(false)));
    if (has_relative_attention_bias_) {
      relative_attention_bias_ = register_module(
          "relative_attention_bias",
          torch::nn::Embedding(relative_attention_num_buckets_, n_heads_));
    }
    dropout_layer_ = register_module("dropout", torch::nn::Dropout(dropout_));
  }
  void prune_heads(const std::vector<int64_t>& heads,
                   torch::ScalarType dtype = torch::kBFloat16) {
    if (heads.empty()) return;

    auto [new_heads, indices] = find_pruneable_heads_and_indices(
        heads, n_heads_, key_value_proj_dim_, pruned_heads_, dtype);
    if (new_heads.empty()) return;
    q_ = prune_linear_layer(q_, indices);
    k_ = prune_linear_layer(k_, indices);
    v_ = prune_linear_layer(v_, indices);
    o_ = prune_linear_layer(o_, indices, 1);
    n_heads_ -= new_heads.size();
    inner_dim_ = key_value_proj_dim_ * n_heads_;
    for (int64_t h : new_heads) {
      pruned_heads_.insert(h);
    }
  }
  torch::Tensor _relative_position_bucket(torch::Tensor& relative_position,
                                          bool bidirectional = true,
                                          int64_t num_buckets = 32,
                                          int64_t max_distance = 128) const {
    torch::Tensor relative_buckets =
        torch::zeros_like(relative_position, torch::kLong);
    if (bidirectional) {
      num_buckets /= 2;
      relative_buckets +=
          (relative_position > 0).to(torch::kLong) * num_buckets;
      auto abs_relative_position = torch::abs(relative_position);
      relative_position = abs_relative_position;
    } else {
      relative_position =
          -torch::min(relative_position, torch::zeros_like(relative_position));
    }
    int64_t max_exact = num_buckets / 2;
    torch::Tensor is_small = relative_position < max_exact;
    auto relative_position_float = relative_position.to(torch::kFloat);
    auto max_exact_float = static_cast<float>(max_exact);
    auto max_distance_float = static_cast<float>(max_distance);
    torch::Tensor relative_position_if_large =
        max_exact + (torch::log(relative_position_float / max_exact_float) /
                     std::log(max_distance_float / max_exact_float) *
                     (num_buckets - max_exact))
                        .to(torch::kLong);
    relative_position_if_large = torch::min(
        relative_position_if_large,
        torch::full_like(
            relative_position_if_large, num_buckets - 1, torch::kLong));
    relative_buckets +=
        torch::where(is_small, relative_position, relative_position_if_large);
    return relative_buckets;
  }
  torch::Tensor compute_bias(
      int64_t query_length,
      int64_t key_length,
      c10::optional<torch::Device> device = c10::nullopt,
      const c10::optional<torch::Tensor>& cache_position = c10::nullopt) const {
    if (!has_relative_attention_bias_) {
      return torch::zeros(
          {1, n_heads_, query_length, key_length},
          torch::dtype(torch::kFloat).device(device.value_or(torch::kCPU)));
    }

    torch::Device dev =
        device.value_or(relative_attention_bias_->weight.device());

    torch::Tensor context_position;
    if (cache_position.has_value()) {
      context_position = cache_position.value().unsqueeze(1).to(dev);
    } else {
      context_position =
          torch::arange(query_length, torch::dtype(torch::kLong).device(dev))
              .unsqueeze(1);
    }

    torch::Tensor memory_position =
        torch::arange(key_length, torch::dtype(torch::kLong).device(dev))
            .unsqueeze(0);
    torch::Tensor relative_position = memory_position - context_position;
    torch::Tensor relative_position_bucket =
        _relative_position_bucket(relative_position,
                                  true,
                                  relative_attention_num_buckets_,
                                  relative_attention_max_distance_);
    torch::Tensor values =
        const_cast<torch::nn::EmbeddingImpl*>(relative_attention_bias_.get())
            ->forward(relative_position_bucket);
    values = values.permute({2, 0, 1}).unsqueeze(0);

    return values;
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const c10::optional<torch::Tensor>& mask = c10::nullopt,
      const c10::optional<torch::Tensor>& key_value_states = c10::nullopt,
      const c10::optional<torch::Tensor>& position_bias = c10::nullopt,
      const c10::optional<torch::Tensor>& layer_head_mask = c10::nullopt,
      bool output_attentions = false) {
    int64_t batch_size = hidden_states.size(0);
    int64_t seq_length = hidden_states.size(1);
    bool is_cross_attention = key_value_states.has_value();
    torch::Tensor query_states = q_->forward(hidden_states);
    query_states =
        query_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, seq_len, head_dim)

    torch::Tensor current_states =
        is_cross_attention ? key_value_states.value() : hidden_states;
    torch::Tensor key_states = k_->forward(current_states);
    torch::Tensor value_states = v_->forward(current_states);
    key_states =
        key_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    value_states =
        value_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    torch::Tensor scores = torch::matmul(
        query_states,
        key_states.transpose(3, 2));  // (batch, n_heads, seq_len, key_len)
    torch::Tensor curr_position_bias;
    if (position_bias.has_value() && position_bias.value().numel() > 0) {
      curr_position_bias = position_bias.value();
    } else {
      int64_t key_length = key_states.size(-2);
      if (!has_relative_attention_bias_) {
        curr_position_bias =
            torch::zeros({1, n_heads_, seq_length, key_length},
                         torch::dtype(scores.dtype()).device(scores.device()));
      } else {
        torch::Tensor bias =
            compute_bias(seq_length, key_length, scores.device());
        curr_position_bias = bias.index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(-seq_length, torch::indexing::None),
             torch::indexing::Slice()});
      }
      if (mask.has_value() && mask.value().numel() > 0) {
        torch::Tensor causal_mask = mask.value().index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(0, key_states.size(-2))});
        curr_position_bias = curr_position_bias + causal_mask;
      }
    }
    if (!pruned_heads_.empty()) {
      torch::Tensor head_mask =
          torch::ones(n_heads_ + pruned_heads_.size(), torch::kBool)
              .to(scores.device());
      for (int64_t pruned : pruned_heads_) {
        head_mask[pruned] = false;
      }
      curr_position_bias = curr_position_bias.index({torch::indexing::Slice(),
                                                     head_mask,
                                                     torch::indexing::Slice(),
                                                     torch::indexing::Slice()});
    }
    scores += curr_position_bias;
    torch::Tensor attn_weights =
        torch::softmax(scores.to(torch::kFloat), -1).to(scores.dtype());
    attn_weights = dropout_layer_->forward(attn_weights);
    if (layer_head_mask.has_value() && layer_head_mask.value().numel() > 0) {
      attn_weights = attn_weights * layer_head_mask.value();
    }
    torch::Tensor attn_output = torch::matmul(
        attn_weights, value_states);  // (batch, n_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2)
                      .contiguous();  // (batch, seq_len, n_heads, head_dim)
    attn_output = attn_output.view({batch_size, -1, inner_dim_});
    attn_output = o_->forward(attn_output);
    std::vector<torch::Tensor> outputs = {attn_output, curr_position_bias};
    if (output_attentions) {
      outputs.push_back(attn_weights);
    }
    return outputs;
  }
  void load_state_dict(const StateDict& state_dict) {
    auto q_weight = state_dict.get_tensor("q.weight");
    if (q_weight.defined()) {
      DCHECK_EQ(q_->weight.sizes(), q_weight.sizes())
          << "q weight size mismatch: expected " << q_->weight.sizes()
          << " but got " << q_weight.sizes();
      q_->weight.data().copy_(q_weight);
    }
    auto k_weight = state_dict.get_tensor("k.weight");
    if (k_weight.defined()) {
      DCHECK_EQ(k_->weight.sizes(), k_weight.sizes())
          << "k weight size mismatch: expected " << k_->weight.sizes()
          << " but got " << k_weight.sizes();
      k_->weight.data().copy_(k_weight);
    }
    auto v_weight = state_dict.get_tensor("v.weight");
    if (v_weight.defined()) {
      DCHECK_EQ(v_->weight.sizes(), v_weight.sizes())
          << "v weight size mismatch: expected " << v_->weight.sizes()
          << " but got " << v_weight.sizes();
      v_->weight.data().copy_(v_weight);
    }
    auto o_weight = state_dict.get_tensor("o.weight");
    if (o_weight.defined()) {
      DCHECK_EQ(o_->weight.sizes(), o_weight.sizes())
          << "o weight size mismatch: expected " << o_->weight.sizes()
          << " but got " << o_weight.sizes();
      o_->weight.data().copy_(o_weight);
    }
    auto relative_attention_bias_weight_ =
        state_dict.get_tensor("relative_attention_bias.weight");
    if (relative_attention_bias_weight_.defined()) {
      DCHECK_EQ(relative_attention_bias_->weight.sizes(),
                relative_attention_bias_weight_.sizes())
          << "relative_attention_bias weight size mismatch: expected "
          << relative_attention_bias_->weight.sizes() << " but got "
          << relative_attention_bias_weight_.sizes();
      relative_attention_bias_->weight.data().copy_(
          relative_attention_bias_weight_);
    }
  }

 private:
  bool has_relative_attention_bias_;
  int64_t relative_attention_num_buckets_;
  int64_t relative_attention_max_distance_;
  int64_t d_model_;
  int64_t key_value_proj_dim_;
  int64_t n_heads_;
  double dropout_;
  int64_t inner_dim_;
  c10::optional<int64_t> layer_idx_;
  torch::nn::Linear q_{nullptr};
  torch::nn::Linear k_{nullptr};
  torch::nn::Linear v_{nullptr};
  torch::nn::Linear o_{nullptr};
  torch::nn::Embedding relative_attention_bias_{nullptr};
  torch::nn::Dropout dropout_layer_{nullptr};
  std::unordered_set<int64_t> pruned_heads_;
  torch::Device device_ = torch::kCPU;          // Device for the module
  torch::ScalarType dtype_ = torch::kBFloat16;  // Default dtype
};
TORCH_MODULE(T5Attention);
class T5LayerSelfAttentionImpl : public torch::nn::Module {
 public:
  T5LayerSelfAttentionImpl(int64_t d_model,
                           int64_t d_kv,
                           int64_t num_heads,
                           double dropout_rate,
                           bool has_relative_attention_bias,
                           int64_t relative_attention_num_buckets,
                           int64_t relative_attention_max_distance,
                           double layer_norm_epsilon,
                           c10::optional<int64_t> layer_idx = c10::nullopt,
                           torch::Device device = torch::kCPU,
                           torch::ScalarType dtype = torch::kBFloat16)
      : device_(device), dtype_(dtype) {
    self_attention_ =
        register_module("SelfAttention",
                        T5Attention(d_model,
                                    d_kv,
                                    num_heads,
                                    dropout_rate,
                                    has_relative_attention_bias,
                                    relative_attention_num_buckets,
                                    relative_attention_max_distance,
                                    layer_idx,
                                    device_,
                                    dtype_));
    layer_norm_ = register_module(
        "layer_norm",
        T5LayerNorm(d_model, layer_norm_epsilon, device_, dtype_));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout_rate));
  }
  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("SelfAttention."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }
  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const c10::optional<torch::Tensor>& attention_mask = c10::nullopt,
      const c10::optional<torch::Tensor>& position_bias = c10::nullopt,
      const c10::optional<torch::Tensor>& layer_head_mask = c10::nullopt,
      bool output_attentions = false) {
    torch::Tensor normed_hidden_states = layer_norm_->forward(hidden_states);
    auto attention_output = self_attention_->forward(normed_hidden_states,
                                                     attention_mask,
                                                     c10::nullopt,
                                                     position_bias,
                                                     layer_head_mask,
                                                     output_attentions);
    torch::Tensor updated_hidden_states =
        hidden_states + dropout_->forward(attention_output[0]);
    // hidden_states, position_bias, [attn_weights])
    std::vector<torch::Tensor> outputs = {updated_hidden_states};
    outputs.push_back(attention_output[1]);
    if (output_attentions && attention_output.size() > 2) {
      outputs.push_back(attention_output[2]);
    }
    return outputs;
  }

 private:
  T5Attention self_attention_{nullptr};
  T5LayerNorm layer_norm_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  torch::Device device_ = torch::kCPU;          // Device for the module
  torch::ScalarType dtype_ = torch::kBFloat16;  // Default dtype for the module
};
TORCH_MODULE(T5LayerSelfAttention);
class T5BlockImpl : public torch::nn::Module {
 public:
  T5BlockImpl(int64_t d_model,
              int64_t d_kv,
              int64_t num_heads,
              int64_t d_ff,
              double dropout_rate,
              const std::string& dense_act_fn,
              bool is_gated_act,
              bool has_relative_attention_bias,
              int64_t relative_attention_num_buckets,
              int64_t relative_attention_max_distance,
              double layer_norm_epsilon,
              c10::optional<int64_t> layer_idx = c10::nullopt,
              torch::Device device = torch::kCPU,
              torch::ScalarType dtype = torch::kBFloat16)
      : device_(device), dtype_(dtype) {
    self_attention_ =
        register_module("SelfAttention",
                        T5LayerSelfAttention(d_model,
                                             d_kv,
                                             num_heads,
                                             dropout_rate,
                                             has_relative_attention_bias,
                                             relative_attention_num_buckets,
                                             relative_attention_max_distance,
                                             layer_norm_epsilon,
                                             layer_idx,
                                             device_,
                                             dtype_));
    ff_layer_ = register_module("LayerFFN",
                                T5LayerFFN(d_model,
                                           d_ff,
                                           dropout_rate,
                                           dense_act_fn,
                                           is_gated_act,
                                           layer_norm_epsilon,
                                           device_,
                                           dtype_));
  }
  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const c10::optional<torch::Tensor>& attention_mask = c10::nullopt,
      const c10::optional<torch::Tensor>& position_bias = c10::nullopt,
      const c10::optional<torch::Tensor>& layer_head_mask = c10::nullopt,
      bool output_attentions = false) {
    auto self_attention_outputs =
        self_attention_->forward(hidden_states.to(device_),
                                 attention_mask,
                                 position_bias,
                                 layer_head_mask,
                                 output_attentions);
    torch::Tensor curr_hidden_states = self_attention_outputs[0];
    std::vector<torch::Tensor> attention_outputs;
    for (size_t i = 1; i < self_attention_outputs.size(); ++i) {
      attention_outputs.push_back(self_attention_outputs[i]);
    }
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    curr_hidden_states = ff_layer_->forward(curr_hidden_states);
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    std::vector<torch::Tensor> outputs = {curr_hidden_states};
    outputs.insert(
        outputs.end(), attention_outputs.begin(), attention_outputs.end());
    return outputs;
  }
  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("layer.0."));
    ff_layer_->load_state_dict(state_dict.get_dict_with_prefix("layer.1."));
  }

 private:
  torch::Tensor clamp_inf_values(const torch::Tensor& x) const {
    float max_val;
    if (x.scalar_type() == torch::kFloat16) {
      max_val = 65504.0f;
    } else if (x.scalar_type() == torch::kFloat32) {
      max_val = std::numeric_limits<float>::max();
    } else if (x.scalar_type() == torch::kBFloat16) {
      max_val = 3.3895313892515355e+38f;
    } else {
      max_val = std::numeric_limits<float>::max();
    }
    torch::Tensor clamp_value =
        torch::where(torch::isinf(x).any(),
                     torch::tensor(max_val - 1000.0f, x.options()),
                     torch::tensor(max_val, x.options()));

    return torch::clamp(x, -clamp_value, clamp_value);
  }
  T5LayerSelfAttention self_attention_{nullptr};
  T5LayerFFN ff_layer_{nullptr};
  torch::Device device_ = torch::kCPU;          // Device for the module
  torch::ScalarType dtype_ = torch::kBFloat16;  // Default dtype for the module
};
TORCH_MODULE(T5Block);
class T5EncoderModelImpl : public torch::nn::Module {
 private:
  T5LayerNorm final_layer_norm_{nullptr};
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<T5Block> blocks_;
  torch::nn::Dropout dropout_{nullptr};
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kBFloat16;  // Default dtype for the model
 public:
  T5EncoderModelImpl(const Context& context,
                     torch::Device device = torch::kCPU,
                     torch::ScalarType dtype = torch::kBFloat16) {
    device_ = device;
    dtype_ = dtype;
    ModelArgs args = context.get_model_args();
    embed_tokens_ = register_module(
        "embed_tokens",
        torch::nn::Embedding(args.t5_vocab_size(), args.t5_d_model()));
    embed_tokens_->to(dtype_);  // Set dtype for embeddings
    for (int64_t i = 0; i < args.t5_num_layers(); ++i) {
      bool has_relative_bias = (i == 0);
      blocks_.push_back(
          register_module("block_" + std::to_string(i),
                          T5Block(args.t5_d_model(),
                                  args.t5_d_kv(),
                                  args.t5_num_heads(),
                                  args.t5_d_ff(),
                                  args.t5_dropout_rate(),
                                  args.t5_dense_act_fn(),
                                  args.t5_is_gated_act(),
                                  has_relative_bias,
                                  args.t5_relative_attention_num_buckets(),
                                  args.t5_relative_attention_max_distance(),
                                  args.t5_layer_norm_epsilon(),
                                  i,
                                  device_,
                                  dtype_)));
      blocks_[i]->to(dtype_);  // Set dtype for each block
    }
    final_layer_norm_ = register_module(
        "final_layer_norm",
        T5LayerNorm(
            args.t5_d_model(), args.t5_layer_norm_epsilon(), device_, dtype_));
    final_layer_norm_->to(dtype_);
    dropout_ =
        register_module("dropout", torch::nn::Dropout(args.t5_dropout_rate()));
  }

  torch::nn::Embedding& get_input_embeddings() { return embed_tokens_; }
  void set_input_embeddings(const torch::nn::Embedding& new_embeddings) {
    embed_tokens_ = new_embeddings;
  }
  torch::Tensor create_text_ids() {
    std::vector<int64_t> non_zero_values = {20730,
                                            7437,
                                            5929,
                                            869,
                                            6,
                                            20330,
                                            53,
                                            609,
                                            18,
                                            19489,
                                            29,
                                            21705,
                                            18,
                                            4084,
                                            3202,
                                            1};

    std::vector<int64_t> text_ids_values;
    text_ids_values.reserve(512);
    text_ids_values.insert(
        text_ids_values.end(), non_zero_values.begin(), non_zero_values.end());
    text_ids_values.insert(
        text_ids_values.end(), 512 - non_zero_values.size(), 0);

    return torch::tensor(text_ids_values, torch::dtype(torch::kLong))
        .view({1, 512})
        .to(device_);
  }
  torch::Tensor forward(torch::Tensor input_ids) {
    LOG(INFO) << "Forwarding T5EncoderModel";
    // prepare input parameters
    // input parameters
    // input_ids
    bool output_hidden_states = false;
    bool output_attentions = false;
    c10::optional<torch::Tensor> attention_mask = c10::nullopt;
    c10::optional<torch::Tensor> head_mask = c10::nullopt;
    torch::Tensor embeddings = embed_tokens_->forward(input_ids);
    auto input_shape = embeddings.sizes();  // (batch_size, seq_length, d_model)
    int64_t batch_size = input_shape[0];
    int64_t seq_length = input_shape[1];
    torch::Tensor causal_mask;
    if (attention_mask.has_value()) {
      causal_mask = attention_mask.value()
                        .to(torch::kFloat)
                        .to(device_)
                        .unsqueeze(1)
                        .unsqueeze(1);
      causal_mask = (1.0 - causal_mask) * std::numeric_limits<float>::lowest();
    } else {
      causal_mask = torch::Tensor();
    }
    torch::Tensor hidden_states = dropout_->forward(embeddings);
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    if (output_hidden_states) {
      all_hidden_states.push_back(hidden_states);
    }
    torch::Tensor position_bias = torch::Tensor();
    for (size_t i = 0; i < blocks_.size(); ++i) {
      torch::Tensor layer_head_mask;
      if (head_mask.has_value()) {
        layer_head_mask =
            head_mask.value().index({torch::tensor((int64_t)i)}).to(device_);
      } else {
        layer_head_mask = torch::Tensor();
      }
      auto layer_outputs = blocks_[i]->forward(hidden_states,
                                               causal_mask,
                                               position_bias,
                                               layer_head_mask,
                                               output_attentions);
      hidden_states = layer_outputs[0];
      position_bias = layer_outputs[1];
      if (output_hidden_states) {
        all_hidden_states.push_back(hidden_states);
      }
      if (output_attentions && layer_outputs.size() > 2) {
        all_attentions.push_back(layer_outputs[2]);
      }

      layer_outputs.clear();
    }
    hidden_states = final_layer_norm_->forward(hidden_states);
    hidden_states = dropout_->forward(hidden_states);
    if (output_hidden_states) {
      all_hidden_states.push_back(hidden_states);
    }
    std::vector<torch::Tensor> outputs = {hidden_states};
    if (output_hidden_states) {
      outputs.push_back(
          torch::stack(all_hidden_states,
                       1));  // (batch_size, num_layers, seq_length, d_model)
    }
    if (output_attentions) {
      outputs.push_back(torch::stack(
          all_attentions,
          1));  // (batch_size, num_layers, n_heads, seq_length, seq_length)
    }
    return outputs[0];
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      const auto embedding_weight = state_dict->get_tensor("shared.weight");
      if (embedding_weight.defined()) {
        DCHECK_EQ(embedding_weight.sizes(), embed_tokens_->weight.sizes())
            << "Embedding weight size mismatch: expected "
            << embed_tokens_->weight.sizes() << ", got "
            << embedding_weight.sizes();
        embed_tokens_->weight.data().copy_(embedding_weight);
      }
      const auto final_layer_norm_weight =
          state_dict->get_tensor("encoder.final_layer_norm.weight");
      if (final_layer_norm_weight.defined()) {
        DCHECK_EQ(final_layer_norm_weight.sizes(),
                  final_layer_norm_->weight.sizes())
            << "Final layer norm weight size mismatch: expected "
            << final_layer_norm_->weight.sizes() << ", got "
            << final_layer_norm_weight.sizes();
        final_layer_norm_->weight.data().copy_(final_layer_norm_weight);
      }
      for (int64_t i = 0; i < blocks_.size(); ++i) {
        const auto block_prefix = "encoder.block." + std::to_string(i) + ".";
        blocks_[i]->load_state_dict(
            state_dict->get_dict_with_prefix(block_prefix));
      }
    }
    LOG(INFO) << "T5EncoderModel loaded successfully.";
  }
};
TORCH_MODULE(T5EncoderModel);
REGISTER_MODEL_ARGS(t5, [&] {
  LOAD_ARG_OR(model_type, "model_type", "t5encoder");
  LOAD_ARG_OR(t5_vocab_size, "vocab_size", 32128);
  LOAD_ARG_OR(t5_d_model, "d_model", 4096);
  LOAD_ARG_OR(t5_num_layers, "num_layers", 24);
  LOAD_ARG_OR(t5_d_kv, "d_kv", 64);
  LOAD_ARG_OR(t5_num_heads, "num_heads", 64);
  LOAD_ARG_OR(t5_d_ff, "d_ff", 10240);
  LOAD_ARG_OR(t5_dropout_rate, "dropout_rate", 0.1f);
  LOAD_ARG_OR(t5_dense_act_fn, "dense_act_fn", "gelu_new");
  LOAD_ARG_OR(t5_is_gated_act, "is_gated_act", true);
  LOAD_ARG_OR(
      t5_relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(t5_relative_attention_max_distance,
              "relative_attention_max_distance",
              128);
  LOAD_ARG_OR(t5_layer_norm_epsilon, "layer_norm_epsilon", 1e-6f);
});
}  // namespace xllm::hf