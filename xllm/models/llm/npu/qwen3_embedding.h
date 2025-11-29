#pragma once

#include <memory>

#include "core/framework/model/embedding_lm.h"
#include "embedding_model_base.h"
#include "qwen3.h"

namespace xllm {

class QWen3ForEmbeddingImpl : public LlmForEmbeddingImplBase<QWen3Model> {
 public:
  QWen3ForEmbeddingImpl(const ModelContext& context)
      : LlmForEmbeddingImplBase<QWen3Model>(context),
        options_(context.get_tensor_options()) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Device device() const { return options_.device(); }
  const torch::TensorOptions& options() const { return options_; }

 private:
  torch::TensorOptions options_;
};
TORCH_MODULE(QWen3ForEmbedding);

template <>
class EmbeddingLMImpl<xllm::QWen3ForEmbedding> : public EmbeddingLM {
 public:
  EmbeddingLMImpl(xllm::QWen3ForEmbedding model,
                  const torch::TensorOptions& options)
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

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return model_->device(); }

  const torch::TensorOptions& options() const override {
    return model_->options();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  // Delegate head/embedding accessors to underlying model implementation.
  layer::LmHead get_lm_head() override { return model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) override { model_->set_lm_head(head); }
  layer::WordEmbedding get_word_embedding() override {
    return model_->get_word_embedding();
  }
  void set_word_embedding(layer::WordEmbedding& embedding) override {
    model_->set_word_embedding(embedding);
  }

 private:
  xllm::QWen3ForEmbedding model_;
  torch::TensorOptions options_;
};

REGISTER_EMBEDDING_MODEL_WITH_VARNAME(qwen3_embedding,
                                      qwen3,
                                      QWen3ForEmbedding);
}  // namespace xllm