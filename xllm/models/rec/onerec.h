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

#include <glog/logging.h>
#include <torch/torch.h>

#include <type_traits>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/word_embedding.h"
#include "models/model_registry.h"
#include "models/rec/onerec_args.h"
#include "models/rec/rec_model_base.h"

namespace xllm {

class OneRecModelImpl : public torch::nn::Module {
 public:
  explicit OneRecModelImpl(const ModelContext& context) {
    hidden_size_ = context.get_model_args().hidden_size();
    options_ = context.get_tensor_options();
    shared_ = register_module("shared", layer::WordEmbedding(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    (void)positions;
    (void)kv_caches;

    if (!tokens.defined()) {
      return ModelOutput();
    }

    if (tokens.numel() == 0) {
      return ModelOutput(torch::empty({0, hidden_size_}, options_));
    }

    if (const auto* onerec_params = input_params.onerec_params()) {
      if (onerec_params->is_hybrid_mode &&
          tokens.scalar_type() != torch::kLong &&
          tokens.scalar_type() != torch::kInt) {
        return ModelOutput(tokens);
      }
    }

    return ModelOutput(shared_(tokens));
  }

  void load_state_dict(const StateDict& state_dict) {
    auto shared_dict = state_dict.get_dict_with_prefix("shared.");
    if (!shared_dict.empty()) {
      shared_->load_state_dict(shared_dict);
    }
  }

  layer::WordEmbedding get_word_embedding() { return shared_; }

  void set_word_embedding(layer::WordEmbedding& embedding) {
    shared_ = embedding;
  }

 private:
  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  layer::WordEmbedding shared_{nullptr};
};
TORCH_MODULE(OneRecModel);

class OneRecForConditionalGenerationImpl
    : public RecForCausalLMImplBase<OneRecModel> {
 public:
  explicit OneRecForConditionalGenerationImpl(const ModelContext& context)
      : RecForCausalLMImplBase<OneRecModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    (void)prefix;
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(*state_dict);

      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("shared."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
  }
};
TORCH_MODULE(OneRecForConditionalGeneration);

using OneRecCausalLM = CausalLMImpl<OneRecForConditionalGeneration>;
static_assert(std::is_base_of_v<CausalLM, OneRecCausalLM>,
              "OneRec must satisfy CausalLM contract.");

REGISTER_REC_MODEL(onerec, OneRecForConditionalGeneration);

}  // namespace xllm
