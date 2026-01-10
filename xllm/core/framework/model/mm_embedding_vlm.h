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

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "causal_vlm.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "model_args.h"
#include "model_input_params.h"

namespace xllm {

class MMEmbeddingVLM : public CausalVLM {
 public:
  ~MMEmbeddingVLM() override = default;
};

template <typename Model>
class MMEmbeddingVLMImpl : public MMEmbeddingVLM {
 public:
  MMEmbeddingVLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  virtual MMDict encode(const ModelInputParams& input_params) override {
    return model_->encode(input_params);
  };

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    return torch::Tensor();
  }

  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& input_params) override {
    return torch::Tensor{};
  }

  virtual torch::Tensor forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    return torch::Tensor{};
  }
  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

#if defined(USE_NPU)
  virtual void set_npu_lm_head(layer::NpuLmHead& head) { return; }
  virtual layer::NpuLmHead get_npu_lm_head() { return nullptr; }
  virtual layer::NpuWordEmbedding get_npu_word_embedding() { return nullptr; }
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    return;
  }
#endif
  virtual void set_lm_head(layer::LmHead& head) { return; }
  virtual layer::LmHead get_lm_head() { return nullptr; }
  virtual layer::WordEmbedding get_word_embedding() { return nullptr; }
  virtual void set_word_embedding(layer::WordEmbedding& embedding) { return; }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
