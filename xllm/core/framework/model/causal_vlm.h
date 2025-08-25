#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "causal_lm.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/parallel_state.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "model_args.h"
#include "model_input_params.h"

namespace xllm {

class CausalVLM : public CausalLM {
 public:
  ~CausalVLM() override = default;
};

template <typename Model>
class CausalVLMImpl : public CausalVLM {
 public:
  CausalVLMImpl(Model model, const torch::TensorOptions& options)
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

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

  hf::LlmHead get_lm_head() override { return model_->get_lm_head(); };

  void set_lm_head(hf::LlmHead& head) override { model_->set_lm_head(head); };

  hf::AtbWordEmbedding get_word_embedding() override {
    return model_->get_word_embedding();
  };

  void set_word_embedding(hf::AtbWordEmbedding& embedding) override {
    model_->set_word_embedding(embedding);
  };

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;
  torch::TensorOptions options_;
};

}  // namespace xllm
