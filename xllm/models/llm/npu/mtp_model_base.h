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

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <typeinfo>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/npu/npu_block_copy_impl.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm {
template <typename DecoderLayerType>
class MtpModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  MtpModelImplBase(const std::string& model_type, const ModelContext& context)
      : model_type_(model_type) {
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });

    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    num_experts_per_tok_ = model_args.num_experts_per_tok();

    embed_tokens_ =
        register_module("embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);

    // MTP extra module
    eh_proj_ =
        register_module("eh_proj", layer::NpuColumnParallelLinear(context));
    enorm_ = register_module("enorm", layer::NpuRMSNorm(context));
    hnorm_ = register_module("hnorm", layer::NpuRMSNorm(context));
    final_norm_ = register_module("final_norm", layer::NpuRMSNorm(context));

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = DecoderLayerType(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual torch::Tensor forward(torch::Tensor tokens,
                                torch::Tensor positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    if (dp_size_ > 1 && tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }

    torch::Tensor h = embed_tokens_(tokens, 0);
    torch::Tensor enorm = enorm_(h, 0);
    torch::Tensor input_embedding = input_params.input_embedding;
    if (input_embedding.defined()) {
      h = input_embedding;
    } else {
      LOG(WARNING) << "hnorm use embedding from tokens.";
    }

    torch::Tensor hnorm = hnorm_(h, 0);
    CHECK_EQ(enorm.dim(), hnorm.dim());
    CHECK_EQ(enorm.size(0), hnorm.size(0));
    h = torch::cat({enorm, hnorm}, /*dim=*/-1);
    h = eh_proj_(h, 0);

    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (model_type_ == "glm4_moe_mtp") {
      cos_pos = cos_pos.view(at::IntArrayRef{-1, 2, cos_pos.size(-1) / 2});
      sin_pos = sin_pos.view(at::IntArrayRef{-1, 2, sin_pos.size(-1) / 2});
    }

    torch::Tensor attn_mask;
    // TODO(liangzhiwei20): support prefix cache for deepseek .
    if (FLAGS_enable_chunked_prefill) {
      int num_sequences = input_params.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);

        for (int j = 0; j < num_sequences; j++) {
          auto mask =
              attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                         input_params.kv_seq_lens_vec[j],
                                         input_params.kv_max_seq_len,
                                         h.dtype().toScalarType(),
                                         h.device());
          req_mask_vec.emplace_back(mask);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    } else {
      attn_mask =
          attn_mask_.get_attn_mask(128, h.dtype().toScalarType(), h.device());
    }

    int64_t input_length = tokens.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));

    // TODO(liangzhiwei20): MTP need more support for layer wise copy.
    if (input_params.layer_wise_load_synchronizer != nullptr) {
      LOG(FATAL) << "MTP not support layer wise copy!";
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    input_params_new.expert_array = expert_array;

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return torch::Tensor();
      }

      auto& layer = layers_[i];

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
        return torch::Tensor();
      }

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);
    }

    return final_norm_(h, 0);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    eh_proj_->load_state_dict(state_dict.get_dict_with_prefix("eh_proj."));
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_head.norm."));
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    eh_proj_->verify_loaded_weights(prefix + "eh_proj.");
    enorm_->verify_loaded_weights(prefix + "enorm.");
    hnorm_->verify_loaded_weights(prefix + "hnorm.");
    final_norm_->verify_loaded_weights(prefix + "shared_head.norm.");
  }

  virtual void merge_loaded_weights() {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    eh_proj_->merge_loaded_weights();
    enorm_->merge_loaded_weights();
    hnorm_->merge_loaded_weights();
    final_norm_->merge_loaded_weights();
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return embed_tokens_;
  }

  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 protected:
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t num_experts_per_tok_;

  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  layer::NpuWordEmbedding embed_tokens_{nullptr};
  layer::AttentionMask attn_mask_;

  // MTP extra modules
  layer::NpuColumnParallelLinear eh_proj_{nullptr};
  layer::NpuRMSNorm enorm_{nullptr};
  layer::NpuRMSNorm hnorm_{nullptr};
  layer::NpuRMSNorm final_norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<DecoderLayerType> layers_;

  bool layer_forward_interrupted_ = false;

 private:
  std::string model_type_;
};

template <typename MtpModelType>
class MtpForCausalLMImplBase : public torch::nn::Module {
 public:
  MtpForCausalLMImplBase(const ModelContext& context) {
    model_ = register_module("model", MtpModelType(context));
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
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  virtual void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }

    // verify
    model_->verify_loaded_weights(prefix);

    model_->merge_loaded_weights();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::NpuLmHead get_npu_lm_head() { return lm_head_; }

  virtual void set_npu_lm_head(layer::NpuLmHead& head) { lm_head_ = head; }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& word_embedding) {
    model_->set_npu_word_embedding(word_embedding);
  }

 protected:
  MtpModelType model_{nullptr};
  layer::NpuLmHead lm_head_{nullptr};
};
}  // namespace xllm