/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
// clang-format on
#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model_loader.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "layers/lm_head.h"
#include "layers/word_embedding.h"
#include "model_args.h"
#include "model_input_params.h"

namespace xllm {

namespace detail {
template <typename T, typename = void>
struct has_get_lm_head : std::false_type {};

template <typename T>
struct has_get_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->get_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_lm_head : std::false_type {};

template <typename T>
struct has_set_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->set_lm_head(
                           std::declval<layer::LmHead&>()))>> : std::true_type {
};

template <typename T, typename = void>
struct has_get_word_embedding : std::false_type {};

template <typename T>
struct has_get_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_word_embedding : std::false_type {};

template <typename T>
struct has_set_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_word_embedding(
        std::declval<layer::WordEmbedding&>()))>> : std::true_type {};
}  // namespace detail

class CausalLM : public torch::nn::Module {
 public:
  ~CausalLM() override = default;

  // tokens: [num_tokens]
  // positions: [num_tokens]
  // returns: [num_tokens, hidden_size]
  virtual torch::Tensor forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;

  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;

  virtual torch::Device device() const = 0;

  virtual void prepare_expert_weight(
      int32_t layer_id,
      const std::vector<int32_t>& expert_ids) = 0;
  virtual void update_expert_weight(int32_t layer_id) = 0;

  virtual const torch::TensorOptions& options() const = 0;

  // MTP-specific interface.
  virtual layer::LmHead get_lm_head() {
    LOG(FATAL)
        << "Method 'get_lm_head' is not implemented/supported by this model.";
  }
  virtual void set_lm_head(layer::LmHead& head) {
    LOG(FATAL)
        << "Method 'set_lm_head' is not implemented/supported by this model.";
  }
  virtual layer::WordEmbedding get_word_embedding() {
    LOG(FATAL) << "Method 'get_word_embedding' is not implemented/supported by "
                  "this model.";
  }
  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    LOG(FATAL) << "Method 'set_word_embedding' is not implemented/supported by "
                  "this model.";
  }
};

template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }
  virtual void prepare_expert_weight(
      int32_t layer_id,
      const std::vector<int32_t>& expert_ids) override {
    return model_->prepare_expert_weight(layer_id, expert_ids);
  }

  virtual void update_expert_weight(int32_t layer_id) {
    return model_->update_expert_weight(layer_id);
  }

  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<Model>::value) {
      return model_->get_lm_head();
    } else {
      return CausalLM::get_lm_head();
    }
  };

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<Model>::value) {
      model_->set_lm_head(head);
    } else {
      CausalLM::set_lm_head(head);
    }
  };

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<Model>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  };

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<Model>::value) {
      model_->set_word_embedding(embedding);
    } else {
      CausalLM::set_word_embedding(embedding);
    }
  };

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
