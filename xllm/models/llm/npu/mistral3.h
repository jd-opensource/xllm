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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/util/timer.h"
#include "llm_model_base.h"
#include "mistral.h"
#include "models/model_registry.h"
namespace xllm {
// Mistral3 model (without LM head)
class Mistral3ModelImpl : public torch::nn::Module {
 public:
  explicit Mistral3ModelImpl(const ModelContext& context) {
    language_model_ = register_module("language_model", MistralModel(context));
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto output =
        language_model_->forward(tokens, positions, kv_caches, input_params);
    return output;  // Return directly, including aux_hidden_states
  }

  void load_state_dict(const StateDict& state_dict) {
    language_model_->load_state_dict(state_dict.get_dict_with_prefix("model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    language_model_->verify_loaded_weights(prefix + "model.");
  }

  // Add lifecycle functions
  void merge_loaded_weights() {
    LOG(INFO) << "Merging loaded weights for Mistral3Model";
    if (language_model_) {
      language_model_->merge_loaded_weights();
    }
  }

  void merge_and_move_pinned_host() {
    if (language_model_) {
      language_model_->merge_and_move_pinned_host();
    }
  }

  void free_weights() {
    if (language_model_) {
      language_model_->free_weights();
    }
  }

  void reload_weights() {
    if (language_model_) {
      language_model_->reload_weights();
    }
  }

  void reload_weights_from_device() {
    if (language_model_) {
      language_model_->reload_weights_from_device();
    }
  }

 private:
  MistralModel language_model_{nullptr};
};
TORCH_MODULE(Mistral3Model);

// Mistral3 model for conditional generation (text-only)
class Mistral3ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Mistral3ForConditionalGenerationImpl(const ModelContext& context) {
    // register submodules
    model_ = register_module("model", Mistral3Model(context));

    lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto model_output = model_(tokens, positions, kv_caches, input_params);
    auto indices = torch::tensor({9, 19, 29}, torch::kLong)
                       .to(model_output.aux_hidden_states.device());
    auto selected =
        model_output.aux_hidden_states.index_select(/*dim=*/0, indices);

    return ModelOutput(selected);
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    // Add FLAGS_enable_return_mm_full_embeddings
    auto h = hidden_states;
    // return full embeddings if set flag
    if (FLAGS_enable_return_mm_full_embeddings) {
      return h;
    }
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.get_dict_with_prefix("language_model."));
    lm_head_->load_state_dict(
        state_dict.get_dict_with_prefix("language_model.lm_head."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights(prefix + "lm_head.");
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "Loading Mistral3ForConditionalGeneration from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix("language_model."));
      lm_head_->load_state_dict(
          state_dict->get_dict_with_prefix("language_model.lm_head."));
    }

    // Critical: add merge_loaded_weights call!
    if (model_) {
      model_->merge_loaded_weights();
    }
    if (lm_head_) {
      lm_head_->merge_loaded_weights();
    }

    model_->verify_loaded_weights("language_model.");
    lm_head_->verify_loaded_weights("language_model.lm_head.");
    LOG(INFO) << "Mistral3ForConditionalGeneration loaded successfully.";
  }

  // Add lifecycle functions
  void merge_loaded_weights() {
    if (model_) {
      model_->merge_loaded_weights();
    }
    if (lm_head_) {
      lm_head_->merge_loaded_weights();
    }
  }

  void merge_and_move_pinned_host() {
    if (model_) {
      model_->merge_and_move_pinned_host();
    }
    if (lm_head_) {
      lm_head_->merge_and_move_pinned_host();
    }
  }

  void free_weights() {
    if (model_) {
      model_->free_weights();
    }
    if (lm_head_) {
      lm_head_->free_weights();
    }
  }

  void reload_weights() {
    if (model_) {
      model_->reload_weights();
    }
    if (lm_head_) {
      lm_head_->reload_weights();
    }
  }

  void reload_weights_from_device() {
    if (model_) {
      model_->reload_weights_from_device();
    }
    if (lm_head_) {
      lm_head_->reload_weights_from_device();
    }
  }

 private:
  // parameter members, must be registered
  Mistral3Model model_{nullptr};
  layer::NpuLmHead lm_head_{nullptr};
};
TORCH_MODULE(Mistral3ForConditionalGeneration);

// Model registration
REGISTER_CAUSAL_MODEL(mistral3, Mistral3ForConditionalGeneration);

REGISTER_MODEL_ARGS(mistral3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mistral3");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 131072);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5120);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 32768);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1e9);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
});

}  // namespace xllm
