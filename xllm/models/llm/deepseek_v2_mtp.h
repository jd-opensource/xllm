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

#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/attention_mask.h"
#include "core/layers/column_parallel_linear.h"
#include "core/layers/deepseek_v2_decoder_layer.h"
#include "core/layers/lm_head.h"
#include "core/layers/pos_embedding.h"
#include "core/layers/rms_norm.h"
#include "core/layers/word_embedding.h"
#include "deepseek_v2.h"
#include "framework/model/model_input_params.h"
#include "models/model_registry.h"

// DeepSeek v2 compatible with huggingface weights
// ref to:
// https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/model_executor/models/deepseek_v2.py

namespace xllm {

class DeepseekV2MtpModelImpl : public torch::nn::Module {
 public:
  DeepseekV2MtpModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    blocks_ = register_module("layers", torch::nn::ModuleList());

    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    auto options = context.get_tensor_options();

    layers_.reserve(model_args.n_layers());

    // rotary positional embedding
    auto inv_freq = rotary::apply_deepseek_yarn_rope_scaling(
        model_args.rope_scaling_factor(),
        model_args.rope_extrapolation_factor(),
        model_args.rope_scaling_beta_fast(),
        model_args.rope_scaling_beta_slow(),
        model_args.rotary_dim(),
        model_args.rope_theta(),
        model_args.rope_scaling_original_max_position_embeddings());
    float sm_scale = 1.0f;
    max_seq_len_ = model_args.max_position_embeddings();
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), /*mask_value=*/1);

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = DeepseekV2DecoderLayer(context, i, sm_scale);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      pos_embs_.push_back(create_rotary_embedding(model_args,
                                                  model_args.rotary_dim(),
                                                  inv_freq,
                                                  /*interleaved=*/false,
                                                  sm_scale,
                                                  options));
      atb_pos_embs_.push_back(layer::PosEmbedding(context));
      eh_projs_.push_back(layer::NpuColumnParallelLinear(context));
    }
    enorm_ = register_module("enorm", layer::NpuRmsNorm(context));
    hnorm_ = register_module("hnorm", layer::NpuRmsNorm(context));
    final_norm_ = register_module("final_norm", layer::NpuRmsNorm(context));

    // dp_size_=4;
    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    mapping_data_ = parallel_args.mapping_data();
    num_experts_per_tok_ = model_args.num_experts_per_tok();
    for (int i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(std::vector<torch::Tensor> tokens,
                        std::vector<torch::Tensor> positions,
                        std::vector<KVCache>& kv_caches,
                        const std::vector<ModelInputParams>& input_params) {
    auto micro_batch_num = tokens.size();
    std::vector<torch::Tensor> hs;
    hs.reserve(micro_batch_num);
    std::vector<torch::Tensor> cos_poss;
    cos_poss.reserve(micro_batch_num);
    std::vector<torch::Tensor> sin_poss;
    sin_poss.reserve(micro_batch_num);
    std::vector<torch::Tensor> attn_masks;
    attn_masks.reserve(micro_batch_num);

    for (auto i = 0; i < micro_batch_num; ++i) {
      if (dp_size_ > 1) {
        if (tokens[i].sizes() == 0) {
          tokens[i] = torch::tensor({1}).to(torch::kInt32).to(device_);
          positions[i] = torch::tensor({0}).to(torch::kInt32).to(device_);
        }
      }

      hs.push_back(std::move(embed_tokens_[i](tokens[i], 0)));
      torch::Tensor enorm = enorm_(hs[i], 0);
      const auto& res = input_params[i].mm_data.get<torch::Tensor>("embedding");
      if (res) {
        hs[i] = res.value();
      } else {
        LOG(WARNING) << "hnorm use embedding from tokens.";
      }

      torch::Tensor hnorm = hnorm_(hs[i], 0);
      CHECK_EQ(enorm.dim(), hnorm.dim());
      CHECK_EQ(enorm.size(0), hnorm.size(0));
      hs[i] = torch::cat({enorm, hnorm}, /*dim=*/-1);
      hs[i] = eh_projs_[i](hs[i], 0);

      auto cos_sin =
          atb_pos_embs_[i](pos_embs_[i]->get_cos_sin_cache(), positions[i], 0);
      auto cos_sin_chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      auto cos_pos = cos_sin_chunks[0].contiguous();
      auto sin_pos = cos_sin_chunks[1].contiguous();

      auto attn_mask = attn_mask_.get_attn_mask(
          128, cos_pos.dtype().toScalarType(), cos_pos.device());
      cos_poss.push_back(std::move(cos_pos));
      sin_poss.push_back(std::move(sin_pos));
      attn_masks.push_back(std::move(attn_mask));
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      std::vector<aclrtEvent*> events(micro_batch_num, nullptr);
      std::vector<std::atomic<bool>*> event_flags(micro_batch_num, nullptr);
      for (auto j = 0; j < micro_batch_num; ++j) {
        if (input_params[j].layer_synchronizer != nullptr) {
          events[j] = input_params[j].layer_synchronizer->get_event(i);
          event_flags[j] =
              input_params[j].layer_synchronizer->get_event_flag(i);
        }
      }
      auto& layer = layers_[i];
      layer(hs,
            cos_poss,
            sin_poss,
            attn_masks,
            kv_caches[i],
            input_params,
            events,
            event_flags);
    }
    auto cancated_h = torch::cat(hs, 0);
    return final_norm_(cancated_h, 0);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      eh_projs_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("eh_proj."));
    }
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_head.norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      eh_projs_[i]->verify_loaded_weights(prefix + "eh_proj.");
    }
    enorm_->verify_loaded_weights(prefix + "enorm.");
    hnorm_->verify_loaded_weights(prefix + "hnorm.");
    final_norm_->verify_loaded_weights(prefix + "shared_head.norm.");
  }

  void merge_loaded_weights() {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      eh_projs_[i]->merge_loaded_weights();
    }
    enorm_->merge_loaded_weights();
    hnorm_->merge_loaded_weights();
    final_norm_->merge_loaded_weights();
  }

  std::vector<layer::NpuWordEmbedding> get_word_embedding() {
    return embed_tokens_;
  }

  void set_word_embedding(
      std::vector<layer::NpuWordEmbedding>& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<DeepseekV2DecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  nlohmann::json mapping_data_;
  int32_t num_experts_per_tok_;
  at::Device device_;
  std::vector<layer::NpuWordEmbedding> embed_tokens_;
  std::vector<std::shared_ptr<RotaryEmbedding>> pos_embs_;
  std::vector<layer::PosEmbedding> atb_pos_embs_;
  layer::AttentionMask attn_mask_;
  std::vector<layer::NpuColumnParallelLinear> eh_projs_;
  layer::NpuRmsNorm enorm_{nullptr};
  layer::NpuRmsNorm hnorm_{nullptr};
  layer::NpuRmsNorm final_norm_{nullptr};
};
TORCH_MODULE(DeepseekV2MtpModel);

class DeepseekV2MtpForCausalLMImpl : public torch::nn::Module {
 public:
  DeepseekV2MtpForCausalLMImpl(const ModelContext& context) {
    model_ = register_module("model", DeepseekV2MtpModel(context));
    // lm_head_ = register_module(
    //     "lm_head", layer::LmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const std::vector<torch::Tensor>& tokens,
                        const std::vector<torch::Tensor>& positions,
                        std::vector<KVCache>& kv_caches,
                        const std::vector<ModelInputParams>& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  // load model
  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("model."));
      // lm_head_->load_state_dict(state_dict.get_dict_with_prefix("model.shared_head.head."));
    }

    // verify
    model_->verify_loaded_weights("model.");
    // lm_head_->verify_loaded_weights("model.shared_head.head.");

    model_->merge_loaded_weights();
    // lm_head_->merge_loaded_weights();
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    return;
  }
  void update_expert_weight(int32_t layer_id) { return; }
#if defined(USE_NPU)
  layer::NpuLmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::NpuLmHead& head) { lm_head_ = head; }

  std::vector<layer::NpuWordEmbedding> get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(
      std::vector<layer::NpuWordEmbedding>& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 private:
  layer::NpuLmHead lm_head_{nullptr};
#endif
 private:
  DeepseekV2MtpModel model_{nullptr};
};
TORCH_MODULE(DeepseekV2MtpForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v3_mtp, DeepseekV2MtpForCausalLM);

// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
REGISTER_MODEL_ARGS(deepseek_v3_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v3_mtp");
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
});
}  // namespace xllm