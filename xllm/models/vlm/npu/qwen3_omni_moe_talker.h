#pragma once

#if defined(USE_NPU)
#include <atb/atb_infer.h>

#include "xllm_kernels/core/include/atb_speed/log.h"
#endif

#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>  //TODO: @pxy what are they
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"

// debug
#include "core/util/tensor_helper.h"

namespace xllm {
class Qwen3_Omni_MoeTalkerResizeMLPImpl : public torch::nn::Module {
 public:
  Qwen3_Omni_MoeTalkerResizeMLPImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    // TODO(panxuanyu): to fully support omni
    // linear_fc1 = register_module(
    //     "linear_fc1",
    //     torch::nn::Linear(
    //         torch::nn::LinearOptions(model_args.thinker_hidden_size(),
    //                                  model_args.talker_text_intermediate_size())
    //             .bias(true)));
    // linear_fc2 = register_module(
    //     "linear_fc2",
    //     torch::nn::Linear(
    //         torch::nn::LinearOptions(model_args.talker_text_intermediate_size(),
    //                                  model_args.talker_text_hidden_size())
    //             .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return linear_fc2(torch::silu(linear_fc1(x)));
  }

 private:
  torch::nn::Linear linear_fc1 = nullptr;
  torch::nn::Linear linear_fc2 = nullptr;
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerResizeMLP);

class Qwen3_Omni_MoeTalkerForConditionalGenerationImpl
    : public torch::nn::Module {
 public:
  Qwen3_Omni_MoeTalkerForConditionalGenerationImpl(
      const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    // language model
    model_ = register_module("model", Qwen3MoeForCausalLM(context));

    text_projection_ = register_module("text_projection",
                                       Qwen3_Omni_MoeTalkerResizeMLP(context));
    hidden_projection_ = register_module(
        "hidden_projection_", Qwen3_Omni_MoeTalkerResizeMLP(context));
    // code_predictor_ = register_module();     // TODO(panxuanyu): to fully
    // support omni
  }

  ModelOutput forward(const torch::Tensor& tokens,  // no need
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto hidden_states = model_(tokens, positions, kv_caches, input_params);
    torch::Tensor output =
        hidden_states.hidden_states.flatten().to(torch::kFloat).to(torch::kCPU);
    return ModelOutput(hidden_states);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return model_->logits(hidden_states, seleted_idxes);
  }

  // TODO(panxuanyu): to fully support omni
  //   ModelInputParams prepare_talker_input(){
  //     /*
  //     input: thinker_result
  //     immediate: thinker_embed, thinker_hidden, im_start_indexes,
  //     multimodal_mask, talker_special_tokens, tts_bos_embed, tts_eos_embed,
  //     tts_pad_embed output: talker_input_embed, talker_input_id
  //     */
  //     ModelInputParams input_params;
  //     return input_params;
  //   }

  void load_model(std::unique_ptr<ModelLoader> loader) {  // TODO: @pxy
    LOG(INFO) << "Talker starts to load weight";
    for (auto& state_dict : loader->get_state_dicts()) {
      if (state_dict->get_tensor("talker.codec_head.weight").defined()) {
        state_dict->rename_prefix_inplace("talker.codec_head.",
                                          "talker.model.lm_head.");
      }
    }
    model_->load_model(std::move(loader), "talker.model.");
    LOG(INFO) << "Talker finished loading weight";
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

  layer::NpuLmHead get_npu_lm_head() { return model_->get_npu_lm_head(); }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  Qwen3MoeForCausalLM model_ = nullptr;
  Qwen3_Omni_MoeTalkerResizeMLP text_projection_ = nullptr;
  Qwen3_Omni_MoeTalkerResizeMLP hidden_projection_ = nullptr;
  torch::nn::Linear codec_head_ = nullptr;
  // TODO(panxuanyu): to fully support omni
  // Qwen3_Omni_MoeTalkerCodePredictorModelForConditionalGeneration
  // code_predictor_;
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerForConditionalGeneration);

//**********
// TODO(panxuanyu): remove later
// talker cannot run as single model, this is for debug only
//**********
REGISTER_CAUSAL_MODEL(qwen3_omni_moe_talker,
                      Qwen3_Omni_MoeTalkerForConditionalGeneration);
// loader
REGISTER_MODEL_ARGS(qwen3_omni_moe_talker, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe_talker");
  // Talker basic
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 48);
  // Talker basic architecture
  LOAD_ARG_OR(n_layers, "talker_config.text_config.num_hidden_layers", 20);
  LOAD_ARG(rope_scaling_mrope_section,
           "talker_config.text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(
      attention_bias, "talker_config.text_config.attention_bias", false);
  LOAD_ARG_OR(
      attention_dropout, "talker_config.text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(
      decoder_sparse_step, "talker_config.text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(hidden_act, "talker_config.text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "talker_config.text_config.hidden_size", 1024);
  LOAD_ARG_OR(
      initializer_range, "talker_config.text_config.initializer_range", 0.02f);
  LOAD_ARG_OR(
      intermediate_size, "talker_config.text_config.intermediate_size", 2048);
  LOAD_ARG_OR(max_position_embeddings,
              "talker_config.text_config.max_position_embeddings",
              65536);
  LOAD_ARG_OR(moe_intermediate_size,
              "talker_config.text_config.moe_intermediate_size",
              384);
  LOAD_ARG_OR(norm_topk_prob, "talker_config.text_config.norm_topk_prob", true);
  LOAD_ARG_OR(output_router_logits,
              "talker_config.text_config.output_router_logits",
              false);
  LOAD_ARG_OR(rms_norm_eps, "talker_config.text_config.rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "talker_config.text_config.rope_theta", 1000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef,
              "talker_config.text_config.router_aux_loss_coef",
              0.001f);
  LOAD_ARG_OR(use_sliding_window,
              "talker_config.text_config.use_sliding_window",
              false);
  LOAD_ARG_OR(tie_word_embeddings,
              "talker_config.text_config.tie_word_embeddings",
              false);
  LOAD_ARG_OR(vocab_size, "talker_config.text_config.vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers,
              "talker_config.text_config.mlp_only_layers",
              std::vector<int>());
  // Tokenizer
  LOAD_ARG_OR(eos_token_id, "talker_config.codec_eos_token_id", 2150);
  LOAD_ARG_OR(bos_token_id, "talker_config.codec_bos_id", 2149);
  /*
    "audio_token_id": 151675,
    "codec_bos_id": 2149,
    "codec_eos_token_id": 2150,
    "codec_nothink_id": 2155,
    "codec_pad_id": 2148,
    "codec_think_bos_id": 2156,
    "codec_think_eos_id": 2157,
  */
  // Attention
  LOAD_ARG_OR(head_dim, "talker_config.text_config.head_dim", 128);
  LOAD_ARG_OR(n_kv_heads, "talker_config.text_config.num_key_value_heads", 2);
  LOAD_ARG_OR(n_heads, "talker_config.text_config.num_attention_heads", 16);
  // MLP
  LOAD_ARG_OR(
      n_shared_experts, "talker_config.text_config.num_shared_experts", 1);
  LOAD_ARG_OR(num_experts, "talker_config.text_config.num_experts", 128);
  LOAD_ARG_OR(
      num_experts_per_tok, "talker_config.text_config.num_experts_per_tok", 6);
  // Talker only
  LOAD_ARG_OR(
      talker_text_vocab_size, "talker_config.text_config.vocab_size", 3072);
  LOAD_ARG_OR(thinker_hidden_size, "talker_config.thinker_hidden_size", 2048);
  LOAD_ARG_OR(talker_text_intermediate_size,
              "talker_config.text_config.intermediate_size",
              2048);
  LOAD_ARG_OR(
      talker_text_hidden_size, "talker_config.text_config.hidden_size", 1024);
});
}  // namespace xllm
