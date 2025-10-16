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

#if defined(USE_NPU)
#include <atb/atb_infer.h>
#endif
#include <gflags/gflags.h>
#include <torch/torch.h>

#include <string>
#include <typeinfo>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/attention_mask.h"
#include "core/layers/block_copy.h"
#include "core/layers/lm_head.h"
#include "core/layers/pos_embedding.h"
#include "core/layers/rms_norm.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "xllm_kernels/core/include/atb_speed/log.h"
#else
#include "core/layers/common/attention.h"
#endif

namespace xllm {

torch::Tensor get_concat_rotary_embedding(int64_t dim,
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
class LlmDecoderLayerImplBase : public torch::nn::Module {
 public:
  LlmDecoderLayerImplBase(const ModelContext& context) {
    // register submodules
    decoder_layer_ = register_module("decoder_layer", DecoderType(context));
#if defined(USE_NPU)
    block_copy_ = register_module("block_copy", layer::BlockCopy(context));
#endif
  }

#if defined(USE_NPU)
  virtual torch::Tensor forward(std::vector<torch::Tensor>& x,
                                std::vector<torch::Tensor>& cos_pos,
                                std::vector<torch::Tensor>& sin_pos,
                                std::vector<torch::Tensor>& attn_mask,
                                KVCache& kv_cache,
                                std::vector<ModelInputParams>& input_params,
                                int node_id,
                                std::vector<aclrtEvent*> event,
                                std::vector<std::atomic<bool>*> event_flag) {
#if defined(USE_NPU)
    auto micro_batch_num = x.size();
    for (auto i = 0; i < micro_batch_num; ++i) {
      if (input_params[i].src_block_indices.numel() > 0) {
        block_copy_(kv_cache.get_k_cache(),
                    kv_cache.get_v_cache(),
                    input_params[i].src_block_indices,
                    input_params[i].dst_block_indices,
                    input_params[i].cum_sum,
                    0);
      }
    }
#endif
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag,
                          node_id);
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
  }
  virtual void merge_loaded_weights() {
    decoder_layer_->merge_loaded_weights();
    block_copy_->merge_loaded_weights();
  }
#else
  virtual torch::Tensor forward(torch::Tensor& x,
                                torch::Tensor& positions,
                                const layer::AttentionMetadata& attn_metadata,
                                KVCache& kv_cache,
                                const ModelInputParams& input_params) {
    return decoder_layer_(x, positions, attn_metadata, kv_cache, input_params);
  }
#endif

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

 private:
  DecoderType decoder_layer_{nullptr};
#if defined(USE_NPU)
  layer::BlockCopy block_copy_{nullptr};
#endif
};

template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  LlmModelImplBase(const std::string& model_type, const ModelArgs& args) {
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
#if defined(USE_NPU)
    return embed_tokens_[0](input_ids, 0);
#else
    return embed_tokens_[0](input_ids);
#endif
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual torch::Tensor forward(
      std::vector<torch::Tensor> tokens,
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
    std::vector<ModelInputParams>& input_params_news =
        const_cast<std::vector<ModelInputParams>&>(input_params);

    for (auto i = 0; i < micro_batch_num; ++i) {
      if (tokens[i].numel() == 0) {
        tokens[i] = torch::tensor({1}).to(torch::kInt32).to(tokens[0].device());
        positions[i] =
            torch::tensor({0}).to(torch::kInt32).to(tokens[0].device());
      }
      auto inputs_embeds = input_params[i].input_embedding;
      // test
      torch::Tensor h;
      if (inputs_embeds.defined()) {
        h = inputs_embeds;
      } else {
#if defined(USE_NPU)
        h = embed_tokens_[i](tokens[i], 0);
#else
        h = embed_tokens_[i](tokens[i]);
#endif
      }
      hs.push_back(std::move(h));
#if defined(USE_NPU)
      auto target_cos_sin = atb_pos_embeds_[i](cos_sin_, positions[i], 0);
      auto target_cos_sin_chunks =
          target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      auto cos_pos = target_cos_sin_chunks[0].contiguous();
      auto sin_pos = target_cos_sin_chunks[1].contiguous();

      if (positions[i].dim() == 2) {  // mrope
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
            {positions[i].sizes().front(), -1, cos_pos.sizes().back()}));
        sin_pos = apply(sin_pos.reshape(
            {positions[i].sizes().front(), -1, sin_pos.sizes().back()}));
      }

      torch::Tensor attn_mask;
      if (model_type_ == "qwen2") {
        max_seq_len_ =
            FLAGS_enable_chunked_prefill
                ? std::max(input_params[i].kv_max_seq_len, max_seq_len_)
                : 128;
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
      } else {
        max_seq_len_ =
            FLAGS_enable_chunked_prefill
                ? std::max(input_params[i].kv_max_seq_len, max_seq_len_)
                : 128;
        if (FLAGS_enable_chunked_prefill) {
          int num_sequences = input_params[i].num_sequences;
          if (num_sequences > 0) {
            std::vector<torch::Tensor> req_mask_vec;
            req_mask_vec.reserve(num_sequences);

            for (int j = 0; j < num_sequences; j++) {
              auto mask =
                  attn_mask_.gen_append_mask(input_params[i].q_seq_lens_vec[j],
                                             input_params[i].kv_seq_lens_vec[j],
                                             max_seq_len_,
                                             cos_pos.dtype().toScalarType(),
                                             cos_pos.device());
              req_mask_vec.emplace_back(mask);
            }
            attn_mask = torch::cat(req_mask_vec, 0);
          }
        } else {
          attn_mask = attn_mask_.get_attn_mask(
              max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
        }
      }
      cos_poss.push_back(std::move(cos_pos));
      sin_poss.push_back(std::move(sin_pos));
      attn_masks.push_back(std::move(attn_mask));
#endif
    }
#if defined(USE_NPU)
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

      if (layer_forward_interrupted_) {
        VLOG(1) << "Forward interrupted at layer: " << i;
        return torch::Tensor();
      }

      layer(hs,
            cos_poss,
            sin_poss,
            attn_masks,
            kv_caches[i],
            input_params_news,
            i,
            events,
            event_flags);
    }
    auto cancated_h = torch::cat(hs, 0);
    return norm_(cancated_h, 0);
#else
    bool is_prefill = input_params[0].q_max_seq_len > 1;
    auto attn_metadata =
        layer::AttentionMetadata::build(input_params[0], is_prefill);

    torch::Tensor h;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(
          hs[0], positions[0], attn_metadata, kv_caches[i], input_params[0]);
    }
    return norm_(h);
#endif
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("embed_tokens."));
    }
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

#if defined(USE_NPU)
  virtual void verify_loaded_weights(const std::string& prefix) const {
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_[i]->verify_loaded_weights(prefix + "embed_tokens.");
    }
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_[i]->merge_loaded_weights();
    }
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }
#endif

  virtual std::vector<layer::WordEmbedding> get_word_embedding() {
    return embed_tokens_;
  }

  virtual void set_word_embedding(
      std::vector<layer::WordEmbedding>& word_embedding) {
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_[i] = word_embedding[i];
    }
  }

 protected:
#if defined(USE_NPU)
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  torch::Tensor cos_sin_;
  int max_seq_len_ = 0;
  int device_id = 0;
  layer::AttentionMask attn_mask_;
  std::vector<layer::PosEmbedding> atb_pos_embeds_;
#endif

  std::vector<int64_t> mrope_section_;
  // test
  //  ParallelEmbedding embed_tokens_{nullptr};
  std::vector<layer::WordEmbedding> embed_tokens_;
  layer::RmsNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;

  bool layer_forward_interrupted_ = false;

 private:
  std::string model_type_;
};

template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  LlmForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", LlmModelType(context));

#if defined(USE_NPU)
    lm_head_ = register_module("lm_head", layer::LmHead(context));
#else
    // lm_head_ is default to no quantization
    lm_head_ =
        register_module("lm_head",
                        layer::LmHead(context.get_model_args().hidden_size(),
                                      context.get_model_args().vocab_size(),
                                      /*bias=*/false,
                                      /*gather_output=*/true,
                                      QuantArgs{},
                                      context.get_parallel_args(),
                                      context.get_tensor_options()));
#endif
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual torch::Tensor forward(
      const std::vector<torch::Tensor>& tokens,
      const std::vector<torch::Tensor>& positions,
      std::vector<KVCache>& kv_caches,
      const std::vector<ModelInputParams>& input_params) {
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
#if defined(USE_NPU)
    return lm_head_(hidden_states, seleted_idxes, 0);
#else
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
#endif
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model." /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
#if defined(USE_NPU)
    // verify
    model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_loaded_weights();
    // test
    lm_head_->merge_loaded_weights();
#endif
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  virtual std::vector<layer::WordEmbedding> get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(
      std::vector<layer::WordEmbedding>& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  // parameter members, must be registered
  LlmModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  // test
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm
