#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <mstx/ms_tools_ext.h>
#include <torch/torch.h>

#include "atb_layers/core/include/atb_speed/log.h"
#include "core/common/global_flags.h"
#include "core/framework/context.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state.h"
#include "core/layers/ascend/attn_mask.h"
#include "core/layers/ascend/llama_decoder_layer.h"
#include "core/layers/ascend/llm_head.h"
#include "core/layers/ascend/rms_norm.h"
#include "core/layers/ascend/word_embedding.h"
#include "core/util/tensor_helper.h"
#include "model_registry.h"

// llama2 model compatible with huggingface weights
namespace xllm::hf {

class LlamaDecoderLayerImpl : public torch::nn::Module {
 public:
  LlamaDecoderLayerImpl(const Context& context) {
    // register submodules
    decoder_layer_ = register_module("decoder_layer", LlamaDecoder(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& work_space,
                        int node_id) {
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          context,
                          work_space,
                          node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {}

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }

 private:
  LlamaDecoder decoder_layer_{nullptr};
};
TORCH_MODULE(LlamaDecoderLayer);

std::tuple<torch::Tensor, torch::Tensor> get_llama_rotary_embedding(
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

class LlamaModelImpl : public torch::nn::Module {
 public:
  LlamaModelImpl(const Context& context) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    // register submodules
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(context.get_model_args().n_layers());
    work_space_ = AtbWorkspace(options.device());
    embed_tokens_ = register_module("embed_tokens", AtbWordEmbedding(context));
    norm_ = register_module("norm", RmsNorm(context));

    std::tie(cos_pos_, sin_pos_) =
        get_llama_rotary_embedding(128,
                                   model_args.max_position_embeddings(),
                                   model_args.rope_theta(),
                                   options);
    // encode_attn_mask_ =
    //   AttentionMaskImpl(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = AttentionMaskImpl(options.device(),
                                   options.dtype().toScalarType(),
                                   /*mask_value=*/mask_value);
    max_seq_len_ = 0;
    atb::Status st = atb::CreateContext(&context_);
    LOG_IF(ERROR, st != 0) << "ContextFactory create atb::Context fail";
    device_id_ = options.device().index();
    void* stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    LOG_IF(ERROR, stream == nullptr) << "get current stream fail";
    // context_->SetExecuteStream(atb_speed::Utils::GetCurrentStream());
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = LlamaDecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    // test
    torch::Tensor h = embed_tokens_(tokens, context_, work_space_, 0);
    // auto h = embed_tokens_(tokens);
    auto cos_pos = cos_pos_.index_select(0, positions);
    auto sin_pos = sin_pos_.index_select(0, positions);
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    // torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
    // max_seq_len_ = std::max(max_of_seq.item<int>(), max_seq_len_);
    torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
    max_seq_len_ = FLAGS_enable_chunked_prefill
                       ? std::max(max_of_seq.item<int>(), max_seq_len_)
                       : 128;
    auto attn_mask = attn_mask_.get_attn_mask(
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
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            context_,
            work_space_,
            i);
    }
    h = norm_(h, context_, work_space_, 0);
    return h;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    // test
    embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  AtbWordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(AtbWordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  atb::Context* context_;
  int max_seq_len_ = 0;
  int device_id_ = 0;
  AtbWorkspace work_space_;
  AttentionMaskImpl attn_mask_;
  AtbWordEmbedding embed_tokens_{nullptr};
  RmsNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<LlamaDecoderLayer> layers_;
};
TORCH_MODULE(LlamaModel);

class LlamaForCausalLMImpl : public torch::nn::Module {
 public:
  LlamaForCausalLMImpl(const Context& context) {
    auto options = context.get_tensor_options();

    // register submodules
    model_ = register_module("model", LlamaModel(context));
    device_id_ = options.device().index();
    work_space_ = AtbWorkspace(options.device());
    atb::Status st = atb::CreateContext(&context_);
    void* stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);
    lm_head_ = register_module("lm_head", LlmHead(context));
  }
  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return lm_head_(hidden_states, seleted_idxes, context_, work_space_, 0);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("model."));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
    }

    // verify
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_loaded_weights();
    lm_head_->merge_loaded_weights();
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    return;
  }
  void update_expert_weight(int32_t layer_id) { return; }

  LlmHead get_lm_head() { return lm_head_; }

  void set_lm_head(LlmHead& head) { lm_head_ = head; }

  AtbWordEmbedding get_word_embedding() { return model_->get_word_embedding(); }

  void set_word_embedding(AtbWordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 private:
  // parameter members, must be registered
  LlamaModel model_{nullptr};
  int device_id_ = 0;
  LlmHead lm_head_{nullptr};
  AtbWorkspace work_space_;
  atb::Context* context_;
};
TORCH_MODULE(LlamaForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);

// register the model args
// example config:
// https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(llama, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llama");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  // decide model type based on vocab size
  LOAD_ARG_OR(vocab_size, "vocab_size", 128256);
  if (args->vocab_size() == 128256) {
    // choose the right chat template
    SET_ARG(model_type, "llama3");

    LOAD_ARG_OR(hidden_size, "hidden_size", 8192);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 28672);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 128000);
    // TODO: support a list of eos token ids
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
    LOAD_ARG_OR(rope_theta, "rope_theta", 500000.0f);
    // load rope scaling parameters
    LOAD_ARG(rope_scaling_rope_type, "rope_scaling.rope_type");
    LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
    LOAD_ARG(rope_scaling_low_freq_factor, "rope_scaling.low_freq_factor");
    LOAD_ARG(rope_scaling_high_freq_factor, "rope_scaling.high_freq_factor");
    LOAD_ARG(rope_scaling_original_max_position_embeddings,
             "rope_scaling.original_max_position_embeddings");
    // stop token ids: "<|eom_id|>", "<|eot_id|>"
    SET_ARG(stop_token_ids, std::unordered_set<int32_t>({128008, 128009}));
  } else {
    // llama 2
    LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
    LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
    // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  }

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace xllm::hf
