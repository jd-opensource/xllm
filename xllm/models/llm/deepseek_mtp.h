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

#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "core/layers/deepseek_v2_decoder_layer.h"
#include "llm_model_base.h"

// DeepSeek v2 compatible with huggingface weights
// ref to:
// https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/model_executor/models/deepseek_v2.py

namespace xllm {

class DeepseekMultiTokenPredictorLayerImpl : public torch::nn::Module {
 public:
  DeepseekMultiTokenPredictorLayerImpl(const ModelContext& context,
                                       const int32_t layer_index)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    // register submodules
    enorm_ = register_module("enorm", layer::RMSNorm(context));
    hnorm_ = register_module("hnorm", layer::RMSNorm(context));
    // no quantization for eh_proj
    eh_proj_ =
        register_module("eh_proj",
                        layer::ReplicatedLinear(model_args_.hidden_size() * 2,
                                                model_args_.hidden_size(),
                                                /*bias=*/false,
                                                /*QuantArgs=*/QuantArgs(),
                                                options));
    mtp_block_ = register_module(
        "mtp_block", layer::DeepseekV2DecoderLayer(context, layer_index));
  }

  torch::Tensor forward(torch::Tensor embed,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    // Layer norm on token inputs
    auto enorm_out = std::get<0>(enorm_(embed));

    torch::Tensor embedding_data = input_params.input_embedding;
    // for dummy data parallel run, we set a empty embedding
    if (attn_metadata.is_dummy) {
      embedding_data = torch::zeros({embed.size(0), model_args_.hidden_size()},
                                    embed.options());
    }
    CHECK(embedding_data.defined())
        << "embedding is not defined in input_params.input_embedding";
    torch::Tensor previous_hidden_states = embedding_data;
    previous_hidden_states = std::get<0>(hnorm_(previous_hidden_states));

    // Concatenate along last dimension and project
    auto concat_emb = torch::cat({enorm_out, previous_hidden_states}, -1);
    auto hidden_states = eh_proj_(concat_emb);

    // Pass through mtp block
    hidden_states = mtp_block_(hidden_states,
                               residual,
                               positions,
                               attn_metadata,
                               kv_cache,
                               input_params);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    eh_proj_->load_state_dict(state_dict.get_dict_with_prefix("eh_proj."));
    mtp_block_->load_state_dict(state_dict);
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

 private:
  layer::RMSNorm enorm_{nullptr};
  layer::RMSNorm hnorm_{nullptr};
  layer::ReplicatedLinear eh_proj_{nullptr};
  layer::DeepseekV2DecoderLayer mtp_block_{nullptr};

  ModelArgs model_args_;
};
TORCH_MODULE(DeepseekMultiTokenPredictorLayer);

class DeepseekMTPModelImpl : public torch::nn::Module {
 public:
  DeepseekMTPModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    // get mtp start and end layer index
    mtp_start_layer_idx_ = model_args.n_layers();
    mtp_end_layer_idx_ =
        mtp_start_layer_idx_ + model_args.num_nextn_predict_layers();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    mtp_layers_.reserve(model_args.num_nextn_predict_layers());

    // create mtp layers
    for (int32_t i = mtp_start_layer_idx_; i < mtp_end_layer_idx_; ++i) {
      auto mtp_layer = DeepseekMultiTokenPredictorLayer(context, i);
      mtp_layers_.push_back(mtp_layer);
      blocks_->push_back(mtp_layer);
    }
    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             context.get_parallel_args(),
                                             options));
    norm_ = register_module("norm", layer::RMSNorm(context));

    // get dp size and rank
    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    for (size_t i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

  // Provide batched signature to satisfy callers that pass vectors
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    // for dp, if tokens is empty, set tokens to 1 and positions to 0
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({1}).to(torch::kInt32).to(device_);
      }
      auto& dp_token_nums = modified_input_params.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }

    auto attn_metadata = layer::AttentionMetadata::build(modified_input_params);
    torch::Tensor hidden_states = embed_tokens_(tokens);
    // Mask out embeddings where positions == 0 (for MTP not needed at pos 0)
    auto mask = (positions == 0);  // bool tensor
    if (mask.any().item<bool>()) {
      // set masked rows to zero
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < mtp_layers_.size(); i++) {
      attn_metadata.plan_info->layer_id = i;
      auto& layer = mtp_layers_[i];
      hidden_states = layer(hidden_states,
                            residual,
                            positions,
                            attn_metadata,
                            kv_caches[i],
                            modified_input_params);
    }
    return std::get<0>(norm_(hidden_states, residual));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each layer's load_state_dict function
    for (int32_t i = 0; i < mtp_layers_.size(); i++) {
      int32_t layer_index = mtp_start_layer_idx_ + i;
      mtp_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_index) + "."));
      // there is only one shared_head.norm for deepseek models, so we load it
      // here
      if (i == mtp_layers_.size() - 1) {
        norm_->load_state_dict(state_dict.get_dict_with_prefix(
            "layers." + std::to_string(layer_index) + ".shared_head.norm."));
      }
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<DeepseekMultiTokenPredictorLayer> mtp_layers_;
  int32_t mtp_start_layer_idx_;
  int32_t mtp_end_layer_idx_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
};
TORCH_MODULE(DeepseekMTPModel);

class DeepseekMTPForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekMTPModel> {
 public:
  DeepseekMTPForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekMTPModel>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    // no need to load lm_head since it shares the same weights with main models
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
  }
};
TORCH_MODULE(DeepseekMTPForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_mtp, DeepseekMTPForCausalLM);

// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
REGISTER_MODEL_ARGS(deepseek_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 0);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));

  // extra parameters for DeepSeek-V3.2-Exp
  // example config:
  // https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json
  // set default value to 0 so as to distinguish from DeepSeek-V3.
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);
});
}  // namespace xllm
