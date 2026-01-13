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

#include <string>
#include <vector>

#include "llm_model_base.h"

namespace xllm {

template <typename DecoderLayerType>
class MtpDecoderLayerImplBase : public torch::nn::Module {
 public:
  MtpDecoderLayerImplBase(const ModelContext& context,
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
    mtp_block_ =
        register_module("mtp_block", DecoderLayerType(context, layer_index));
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
  DecoderLayerType mtp_block_{nullptr};

  ModelArgs model_args_;
};

template <typename DecoderLayerType>
class MtpModelImplBase : public torch::nn::Module {
 public:
  MtpModelImplBase(const ModelContext& context)
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
      auto mtp_layer = DecoderLayerType(context, i);
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
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
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
  std::vector<DecoderLayerType> mtp_layers_;
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
}  // namespace xllm
