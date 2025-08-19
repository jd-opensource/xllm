#pragma once

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <torch/torch.h>

#include <string>
#include <typeinfo>
#include <vector>

#include "atb_layers/core/include/atb_speed/log.h"
#include "core/layers/ascend/attn_mask.h"
#include "core/layers/ascend/rms_norm.h"
// test
#include <mstx/ms_tools_ext.h>

#include "core/common/global_flags.h"
#include "core/framework/context.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/ascend/llm_head.h"
#include "core/layers/ascend/pos_embedding.h"
#include "model_registry.h"

namespace xllm::hf {

std::tuple<torch::Tensor, torch::Tensor> get_qwen_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  // auto inv_freq = 1.0 / torch::pow(10000, torch::arange(0, dim, 2, options) /
  // dim);
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  auto emb = torch::cat({freqs, freqs}, -1);
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  return std::make_tuple(rope_cos, rope_sin);
}

torch::Tensor get_qwen_concat_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  auto emb = torch::cat({freqs, freqs}, -1);
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  std::vector<torch::Tensor> cos_sin{rope_cos, rope_sin};
  return torch::cat(cos_sin, -1);
}

template <typename DecoderType>
class QWenDecoderLayerImplBase : public torch::nn::Module {
 public:
  QWenDecoderLayerImplBase(const Context& context) {
    // register submodules
    decoder_layer_ = register_module("decoder_layer", DecoderType(context));
  }

  virtual torch::Tensor forward(torch::Tensor& x,
                                torch::Tensor& cos_pos,
                                torch::Tensor& sin_pos,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                atb::Context* context,
                                AtbWorkspace& work_space,
                                int node_id,
                                aclrtEvent* event = nullptr,
                                std::atomic<bool>* event_flag = nullptr) {
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          context,
                          work_space,
                          event,
                          event_flag,
                          node_id);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
  }
  virtual void merge_loaded_weights() {
    decoder_layer_->merge_loaded_weights();
  }

 private:
  DecoderType decoder_layer_{nullptr};
};

template <typename DecoderLayerType>
class QWenModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  QWenModelImplBase(const std::string& model_type, const ModelArgs& args)
      : model_type_(model_type) {
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, context_, work_space_, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual torch::Tensor forward(torch::Tensor tokens,
                                torch::Tensor positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    auto inputs_embeds = input_params.input_embedding;
    // test
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens, context_, work_space_, 0);
    }
    // auto h = embed_tokens_(tokens);
    auto target_cos_sin =
        atb_pos_emb_(cos_sin_, positions, context_, work_space_, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto sections = mrope_section_;
        sections.insert(sections.end(), sections.begin(), sections.end());

        auto vec = x.split(sections, -1);
        std::vector<torch::Tensor> selects;
        selects.reserve(vec.size());

        for (int64_t i = 0; i < vec.size(); ++i) {
          auto m = vec[i];
          selects.push_back(m[i % mrope_section_.size()]);
        }
        return torch::cat(selects, -1);
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor attn_mask;
    if (model_type_ == "qwen2") {
      torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
      max_seq_len_ = FLAGS_enable_chunked_prefill
                         ? std::max(max_of_seq.item<int>(), max_seq_len_)
                         : 128;
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
    } else if (model_type_ == "qwen3") {
      torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
      max_seq_len_ = FLAGS_enable_chunked_prefill
                         ? std::max(max_of_seq.item<int>(), max_seq_len_)
                         : 128;
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

      if (FLAGS_enable_chunked_prefill) {
        int batch_size = input_params.q_seq_lens_vec.size();
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(batch_size);

        for (int i = 0; i < batch_size; i++) {
          int start =
              input_params.kv_seq_lens_vec[i] - input_params.q_seq_lens_vec[i];
          int end = input_params.kv_seq_lens_vec[i];

          auto req_mask_slice = attn_mask.slice(0, start, end);
          req_mask_vec.emplace_back(req_mask_slice);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    }
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      auto& layer = layers_[i];

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            context_,
            work_space_,
            i,
            event,
            event_flag);
    }
    h = norm_(h, context_, work_space_, 0);
    return h;
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    // test
    embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  virtual AtbWordEmbedding get_word_embedding() { return embed_tokens_; }

  virtual void set_word_embedding(AtbWordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 protected:
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  torch::Tensor cos_sin_;
  atb::Context* context_;
  int max_seq_len_ = 0;
  int device_id = 0;
  AtbWorkspace work_space_;
  AttentionMaskImpl attn_mask_;
  AtbRotaryEmbedding atb_pos_emb_{nullptr};

  std::vector<int64_t> mrope_section_;
  // test
  //  ParallelEmbedding embed_tokens_{nullptr};
  AtbWordEmbedding embed_tokens_{nullptr};
  RmsNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;

 private:
  std::string model_type_;
};

template <typename QWenModelType>
class QWenForCausalLMImplBase : public torch::nn::Module {
 public:
  QWenForCausalLMImplBase(const Context& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", QWenModelType(context));

    auto options = context.get_tensor_options();
    device_id = options.device().index();
    work_space_ = AtbWorkspace(options.device());
    atb::Status st = atb::CreateContext(&context_);
    void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);
    lm_head_ = register_module("lm_head", LlmHead(context));
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
    // torch::npu::synchronize(device_id);
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    // test
    return lm_head_(hidden_states, seleted_idxes, context_, work_space_, 0);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "" /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix(prefix + "model."));
      if (tie_word_embeddings) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "model.embed_tokens."));
      } else {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "lm_head."));
      }
    }

    // verify
    model_->verify_loaded_weights(prefix + "model.");
    lm_head_->verify_loaded_weights(prefix + "lm_head.");

    model_->merge_loaded_weights();
    // test
    lm_head_->merge_loaded_weights();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual LlmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(LlmHead& head) { lm_head_ = head; }

  virtual AtbWordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(AtbWordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  // parameter members, must be registered
  QWenModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  // test
  LlmHead lm_head_{nullptr};
  AtbWorkspace work_space_;
  atb::Context* context_;
};

}  // namespace xllm::hf
