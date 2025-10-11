#pragma once

#include <memory>

#include "core/framework/model/embedding_lm.h"
#include "qwen3.h"

namespace xllm {

class QWen3ForEmbeddingImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForEmbeddingImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context),
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

}  // namespace xllm

namespace xllm {

template <>
class EmbeddingLMImpl<xllm::QWen3ForEmbedding> : public EmbeddingLM {
 public:
  EmbeddingLMImpl(xllm::QWen3ForEmbedding model,
                  const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  torch::Tensor forward(
      const std::vector<torch::Tensor>& tokens,
      const std::vector<torch::Tensor>& positions,
      std::vector<KVCache>& kv_caches,
      const std::vector<ModelInputParams>& parameters) override {
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
  std::vector<layer::WordEmbedding> get_word_embedding() override {
    return model_->get_word_embedding();
  }
  void set_word_embedding(
      std::vector<layer::WordEmbedding>& embedding) override {
    model_->set_word_embedding(embedding);
  }

 private:
  xllm::QWen3ForEmbedding model_;
  torch::TensorOptions options_;
};

}  // namespace xllm

namespace xllm {

REGISTER_EMBEDDING_MODEL_WITH_VARNAME(qwen3_embedding,
                                      qwen3,
                                      QWen3ForEmbedding);
REGISTER_MODEL_ARGS_WITH_VARNAME(qwen3_embedding, qwen3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For embedding models, we typically don't tie word embeddings
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm