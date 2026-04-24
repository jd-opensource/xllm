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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/model/model_output.h"
#include "core/framework/request/compressor_state_id_manager.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer_impl.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "llm_model_base.h"

namespace xllm {

class DeepSeekV4ModelImpl : public torch::nn::Module {
 public:
  explicit DeepSeekV4ModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    hc_mult_ = model_args.hc_mult();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());

    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             context.get_parallel_args(),
                                             options));
    norm_ = register_module(
        "norm",
        layer::RMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));

    hc_head_ =
        register_module("hc_head",
                        layer::HyperConnectionHead(hc_mult_,
                                                   model_args.hidden_size(),
                                                   model_args.hc_eps(),
                                                   model_args.rms_norm_eps(),
                                                   options));

    const int64_t cached_state_num = FLAGS_max_seqs_per_batch * 2;
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = layer::DeepSeekV4DecoderLayer(context, i, cached_state_num);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  }

  ModelOutput forward_native(torch::Tensor tokens,
                             torch::Tensor positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({1}).to(torch::kInt32).to(device_);
      }
      auto& dp_token_nums = modified_input_params.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla()));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    std::vector<int64_t> batch_to_kv_state =
        modified_input_params.batch_to_kv_state;
    if (batch_to_kv_state.empty()) {
      batch_to_kv_state.resize(attn_metadata.kv_seq_lens.size(0));
      std::iota(batch_to_kv_state.begin(), batch_to_kv_state.end(), 0L);
    }

    torch::Tensor hidden_states = embed_tokens_(tokens);
    hidden_states = hidden_states.unsqueeze(1).repeat({1, hc_mult_, 1});
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states,
                            residual,
                            positions,
                            attn_metadata,
                            kv_caches[i],
                            modified_input_params,
                            batch_to_kv_state,
                            tokens);
    }
    hidden_states = hc_head_(hidden_states);
    auto [h, res] = norm_(hidden_states, std::nullopt);
    return ModelOutput(h, res);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return forward_native(tokens, positions, kv_caches, input_params);
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix("embed."));
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    hc_head_->load_state_dict(state_dict.get_dict_with_prefix("hc_head."));
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::DeepSeekV4DecoderLayer> layers_;
  ModelArgs model_args_;
  int32_t dp_size_ = 1;
  int32_t dp_local_tp_size_ = 1;
  int32_t hc_mult_ = 1;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  layer::HyperConnectionHead hc_head_{nullptr};
};
TORCH_MODULE(DeepSeekV4Model);

class DeepSeekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepSeekV4Model> {
 public:
  explicit DeepSeekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepSeekV4Model>(context) {
    CHECK(!this->tie_word_embeddings)
        << "deepseek_v4 does not support tie_word_embeddings. "
           "Please set it to false.";
    CHECK(!FLAGS_enable_prefix_cache)
        << "deepseek_v4 has not supported enable_prefix_cache yet. "
           "Please disable it.";
    CHECK(!FLAGS_enable_chunked_prefill)
        << "deepseek_v4 has not supported enable_chunked_prefill yet. "
           "Please disable it.";
  }

 private:
  static std::string replace_all(std::string str,
                                 const std::string& from,
                                 const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
    return str;
  }

  static std::string remap_parameter_name(std::string name) {
    name = replace_all(name, "hc_attn_base", "hc_attn_pre.hc_base");
    name = replace_all(name, "hc_attn_fn", "hc_attn_pre.hc_fn");
    name = replace_all(name, "hc_attn_scale", "hc_attn_pre.hc_scale");
    name = replace_all(name, "hc_ffn_base", "hc_ffn_pre.hc_base");
    name = replace_all(name, "hc_ffn_fn", "hc_ffn_pre.hc_fn");
    name = replace_all(name, "hc_ffn_scale", "hc_ffn_pre.hc_scale");

    name = replace_all(name, "hc_head_base", "hc_head.hc_head_base");
    name = replace_all(name, "hc_head_fn", "hc_head.hc_head_fn");
    name = replace_all(name, "hc_head_scale", "hc_head.hc_head_scale");

    name = replace_all(name, "w1.weight", "gate_proj.weight");
    name = replace_all(name, "w3.weight", "up_proj.weight");
    name = replace_all(name, "w2.weight", "down_proj.weight");

    return name;
  }

 public:
  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    static_cast<void>(prefix);
    for (const auto& state_dict : loader->get_state_dicts()) {
      std::unordered_map<std::string, torch::Tensor> remapped_dict;
      for (auto it = state_dict->begin(); it != state_dict->end(); ++it) {
        remapped_dict[remap_parameter_name(it->first)] = it->second;
      }

      StateDict remapped_state_dict(remapped_dict);
      model_->load_state_dict(remapped_state_dict);
      lm_head_->load_state_dict(
          remapped_state_dict.get_dict_with_prefix("head."));
    }
  }
};
TORCH_MODULE(DeepSeekV4ForCausalLM);

REGISTER_CAUSAL_MODEL(deepseek_v4, DeepSeekV4ForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 43);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 6);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sqrtsoftplus");
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(head_dim, "head_dim", 512);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR_FUNC(qk_nope_head_dim, "qk_nope_head_dim", [&] {
    return static_cast<int32_t>(args->head_dim() - args->qk_rope_head_dim());
  });
  SET_ARG(rotary_dim, args->qk_rope_head_dim());
  LOAD_ARG_OR(o_groups, "o_groups", 8);
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(sliding_window, "sliding_window", 128);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 1048576);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);

  SET_ARG(kv_lora_rank, args->head_dim());
  SET_ARG(v_head_dim, args->head_dim());

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(rope_scaling_beta_fast, "rope_scaling.beta_fast", 32);
  LOAD_ARG_OR(rope_scaling_beta_slow, "rope_scaling.beta_slow", 1);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "rope_scaling.original_max_position_embeddings",
              65536);
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(swiglu_limit, "swiglu_limit", 10.0f);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  if (!args->compress_ratios().empty()) {
    CHECK_GE(args->compress_ratios().size(),
             static_cast<size_t>(args->n_layers()))
        << "compress_ratios must cover every DeepSeek V4 layer.";
  }
});

}  // namespace xllm
