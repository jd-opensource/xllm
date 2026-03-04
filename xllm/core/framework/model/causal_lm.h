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
#include "layers/npu/npu_lm_head_impl.h"
#include "layers/npu/npu_word_embedding_impl.h"
#endif
// clang-format on
#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model_loader.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "layers/common/lm_head.h"
#include "layers/common/word_embedding.h"
#include "model_args.h"
#include "model_input_params.h"
#include "model_output.h"
#include "model_traits.h"

namespace xllm {

class CausalLM : public torch::nn::Module {
 public:
  ~CausalLM() override = default;

  // tokens: [num_tokens]
  // positions: [num_tokens]
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& parameters) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

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
#if defined(USE_NPU)
  virtual layer::NpuLmHead get_npu_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual void set_npu_lm_head(layer::NpuLmHead& head) { NOT_IMPLEMENTED(); }
  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    NOT_IMPLEMENTED();
  }
#endif
  virtual layer::LmHead get_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual void set_lm_head(layer::LmHead& head) { NOT_IMPLEMENTED(); }
  virtual layer::WordEmbedding get_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    NOT_IMPLEMENTED();
  }
};

template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
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

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<Model>::value) {
      return model_->get_npu_lm_head();
    } else {
      return CausalLM::get_npu_lm_head();
    }
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    if constexpr (detail::has_set_npu_lm_head<Model>::value) {
      model_->set_npu_lm_head(head);
    } else {
      CausalLM::set_npu_lm_head(head);
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<Model>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return CausalLM::get_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<Model>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      CausalLM::set_npu_word_embedding(embedding);
    }
  }
#endif
  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<Model>::value) {
      return model_->get_lm_head();
    } else {
      return CausalLM::get_lm_head();
    }
  }

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<Model>::value) {
      model_->set_lm_head(head);
    } else {
      CausalLM::set_lm_head(head);
    }
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<Model>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<Model>::value) {
      model_->set_word_embedding(embedding);
    } else {
      CausalLM::set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
