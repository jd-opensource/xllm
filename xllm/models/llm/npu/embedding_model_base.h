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

#include "llm_model_base.h"

namespace xllm {

template <typename LlmModelType>
class LlmForEmbeddingImplBase : public torch::nn::Module {
 public:
  LlmForEmbeddingImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", LlmModelType(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual torch::Tensor forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    LOG(ERROR) << "logits() not implemented for Embedding Model!";
    return torch::empty({0});
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "" /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto sub_dict = state_dict->get_dict_with_prefix(prefix + "model.");
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix(prefix);
      }
      model_->load_state_dict(sub_dict);
    }

    // verify
    model_->verify_loaded_weights(prefix + "model.");
    model_->merge_loaded_weights();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  // parameter members, must be registered
  LlmModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  // test
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm
