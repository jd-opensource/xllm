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
#include <cmath>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "models/model_registry.h"
#include "models/rec/rec_model_base.h"

#if defined(USE_NPU)
#include "core/common/global_flags.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu/npu_onerec_block_layer_impl.h"
#endif

namespace xllm {

#if defined(USE_NPU)
inline torch::Tensor pad_encoder_output(const torch::Tensor& encoder_output,
                                        const ModelInputParams& input_params) {
  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

  const int64_t bs = onerec_params->bs;
  const int64_t hidden_size = encoder_output.size(1);
  const auto& seq_lens = onerec_params->encoder_seq_lens;
  const int64_t max_seq_len = onerec_params->encoder_max_seq_len;

  CHECK_EQ(static_cast<int64_t>(seq_lens.size()), bs)
      << "encoder_seq_lens size mismatch.";

  std::vector<torch::Tensor> seq_list;
  seq_list.reserve(static_cast<size_t>(bs));

  int64_t token_offset = 0;
  for (int64_t i = 0; i < bs; ++i) {
    const int64_t seq_len = seq_lens[i];
    seq_list.emplace_back(encoder_output.narrow(0, token_offset, seq_len));
    token_offset += seq_len;
  }

  auto padded_output = torch::nn::utils::rnn::pad_sequence(
      seq_list, /*batch_first=*/true, /*padding_value=*/0.0);

  if (padded_output.size(1) < max_seq_len) {
    auto extra_padding =
        torch::zeros({bs, max_seq_len - padded_output.size(1), hidden_size},
                     encoder_output.options());
    padded_output = torch::cat({padded_output, extra_padding}, /*dim=*/1);
  }

  return padded_output;
}

inline torch::Tensor compute_onerec_position_bias(
    int64_t query_length,
    int64_t key_length,
    int64_t num_heads,
    bool is_decoder,
    layer::WordEmbedding& position_bias_embedding,
    int64_t num_buckets = 32,
    int64_t max_distance = 128,
    const torch::TensorOptions& options = torch::kFloat32,
    bool is_decode_stage = false,
    const ModelInputParams* input_params = nullptr) {
  auto device = options.device();
  auto dtype = options.dtype();

  int64_t actual_query_length = is_decode_stage ? key_length : query_length;
  if (actual_query_length <= 0) {
    actual_query_length = 1;
  }
  if (key_length <= 0) {
    key_length = 1;
  }

  auto context_position =
      torch::arange(actual_query_length,
                    torch::dtype(torch::kLong).device(device))
          .unsqueeze(1);
  auto memory_position =
      torch::arange(key_length, torch::dtype(torch::kLong).device(device))
          .unsqueeze(0);
  auto relative_position = memory_position - context_position;

  auto relative_buckets = torch::zeros_like(relative_position);

  if (!is_decoder) {
    num_buckets = num_buckets / 2;
    relative_buckets += (relative_position > 0).to(torch::kLong) * num_buckets;
    relative_position = torch::abs(relative_position);
  } else {
    relative_position =
        -torch::min(relative_position, torch::zeros_like(relative_position));
  }

  auto max_exact = num_buckets / 2;
  auto is_small = relative_position < max_exact;
  auto relative_position_if_large =
      max_exact + (torch::log(relative_position.to(torch::kFloat) / max_exact) /
                   std::log(static_cast<double>(max_distance) / max_exact) *
                   (num_buckets - max_exact))
                      .to(torch::kLong);

  relative_position_if_large =
      torch::min(relative_position_if_large,
                 torch::full_like(relative_position_if_large, num_buckets - 1));

  relative_buckets +=
      torch::where(is_small, relative_position, relative_position_if_large);

  auto original_shape = relative_buckets.sizes();
  auto flattened_buckets = relative_buckets.flatten();
  auto values = position_bias_embedding(flattened_buckets);

  if (values.dim() == 2) {
    CHECK_EQ(values.size(0), flattened_buckets.size(0));
    values =
        values.view({original_shape[0], original_shape[1], values.size(1)});
  } else if (values.dim() == 1) {
    values =
        values.unsqueeze(-1).expand({flattened_buckets.size(0), num_heads});
    values = values.view({original_shape[0], original_shape[1], num_heads});
  } else {
    LOG(FATAL) << "Unexpected OneRec position bias dim: " << values.dim();
  }

  if (values.dim() == 3) {
    values = values.permute({2, 0, 1});
  }

  if (is_decode_stage && input_params != nullptr &&
      !input_params->kv_seq_lens_vec.empty()) {
    const int seq_kv_len = input_params->kv_seq_lens_vec[0];
    values = values.slice(1, -1, values.size(1)).slice(2, 0, seq_kv_len);
  } else if (is_decode_stage) {
    values = values.slice(1, -1, values.size(1));
  }

  return values.to(dtype);
}

class OneRecStackImpl : public torch::nn::Module {
 public:
  OneRecStackImpl(const ModelContext& context,
                  bool is_decode,
                  layer::WordEmbedding& embed_tokens) {
    const auto& args = context.get_model_args();
    const auto& options = context.get_tensor_options();

    hidden_size_ = args.hidden_size();
    is_decoder_ = is_decode;
    use_absolute_position_embedding_ = args.use_absolute_position_embedding();
    use_moe_ = args.use_moe() && is_decoder_;
    num_experts_per_tok_ = args.num_experts_per_tok();
    relative_attention_num_buckets_ = args.relative_attention_num_buckets();
    relative_attention_max_distance_ = args.relative_attention_max_distance();
    num_heads_ = is_decode ? args.decoder_n_heads() : args.n_heads();

    embed_tokens_ = embed_tokens;
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_ = register_module("position_bias_embedding",
                                                 layer::WordEmbedding(context));
    }

    norm_ = register_module("final_layer_norm", layer::RMSNorm(context));

    blocks_ = register_module("block", torch::nn::ModuleList());
    const uint32_t num_layers =
        is_decode ? args.n_layers() : args.n_encoder_layers();
    layers_.reserve(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
      auto block = layer::NpuOneRecBlockLayer(context, is_decode, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    (void)options;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params,
                        const torch::Tensor& encoder_output = torch::Tensor()) {
    (void)positions;

    const auto* onerec_params = input_params.onerec_params();
    CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

    torch::Tensor h;
    if (onerec_params->is_hybrid_mode && !is_decoder_) {
      h = tokens;
    } else if (onerec_params->decoder_context_embedding.defined()) {
      if (tokens.numel() == 0) {
        h = onerec_params->decoder_context_embedding;
      } else {
        h = embed_tokens_(tokens);

        auto& context_emb = onerec_params->decoder_context_embedding;
        const int64_t hidden_size = context_emb.size(3);
        const int64_t bs = onerec_params->bs;
        const int64_t group_width = onerec_params->group_width;
        const int64_t context_total_tokens = context_emb.size(2);
        const int64_t token_total_tokens = h.size(0);

        const int64_t bs_group = bs * group_width;
        const int64_t seq_len1 =
            token_total_tokens / std::max<int64_t>(1, bs_group);
        const int64_t seq_len2 = context_total_tokens - seq_len1;

        auto token_embedding_reshaped =
            h.view({bs, group_width, seq_len1, hidden_size});
        context_emb.narrow(2, seq_len2, seq_len1)
            .copy_(token_embedding_reshaped);
        h = context_emb.view({-1, hidden_size}).clone();
      }
      if (!h.is_contiguous()) {
        h = h.contiguous();
      }
    } else {
      h = embed_tokens_(tokens);
    }

    torch::Tensor npu_encoder_output = encoder_output;
    if (npu_encoder_output.defined() &&
        npu_encoder_output.device().type() != h.device().type()) {
      npu_encoder_output = npu_encoder_output.to(h.device());
    }

    const bool is_prefill =
        onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
    auto [query_length, key_length] = compute_sequence_lengths(
        input_params.q_max_seq_len, is_prefill, input_params);

    ModelInputParams input_params_local = input_params;
    auto& mutable_onerec_params = input_params_local.mutable_onerec_params();

    const bool is_decode_stage = is_decoder_ && !is_prefill;
    torch::Tensor effective_attn_mask;
    if (use_absolute_position_embedding_) {
      effective_attn_mask =
          create_moe_attention_mask(query_length, h, is_decoder_);
    } else {
      effective_attn_mask = compute_position_bias_mask(
          query_length, key_length, h, is_decode_stage, input_params);
    }

    auto preprocessed_attn_mask =
        preprocess_attention_mask(effective_attn_mask, h);

    if (mutable_onerec_params.encoder_seq_lens_tensor.defined()) {
      auto flattened_tensor =
          mutable_onerec_params.encoder_seq_lens_tensor.flatten();
      mutable_onerec_params.encoder_seq_lens_tensor =
          flattened_tensor.to(h.device(), torch::kInt).contiguous();
    }

    torch::Tensor expert_array;
    if (use_moe_) {
      const int64_t input_length = h.size(0);
      expert_array = torch::arange(
          0,
          input_length * num_experts_per_tok_,
          torch::TensorOptions().dtype(torch::kInt32).device(h.device()));
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
      if (input_params.layer_synchronizer) {
        input_params.layer_synchronizer->synchronize_layer(i);
      }

      KVCache dummy_kv_cache;
      if (is_decoder_) {
        CHECK_LT(i, kv_caches.size())
            << "OneRec decoder layer kv_cache is missing at layer " << i;
      }
      KVCache& kv_cache_ref = is_decoder_ ? kv_caches[i] : dummy_kv_cache;

      layers_[i]->forward(
          h,
          preprocessed_attn_mask,
          kv_cache_ref,
          input_params_local,
          npu_encoder_output.defined() ? &npu_encoder_output : nullptr,
          static_cast<int>(i),
          nullptr,
          nullptr,
          expert_array);
    }

    std::optional<torch::Tensor> residual = std::nullopt;
    h = std::get<0>(norm_->forward(h, residual));
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto embed_dict = state_dict.get_dict_with_prefix("embed_tokens.");
    if (embed_dict.size() > 0) {
      embed_tokens_->load_state_dict(embed_dict);
    }

    if (!use_absolute_position_embedding_ && position_bias_embedding_) {
      auto pos_bias_dict = state_dict.get_dict_with_prefix(
          "block.0.layer.0.SelfAttention.relative_attention_bias.");
      if (pos_bias_dict.size() > 0) {
        position_bias_embedding_->load_state_dict(pos_bias_dict);
      }
    }

    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("block." + std::to_string(i) + "."));
    }

    norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "block." + std::to_string(i) +
                                        ".");
    }
  }

  void merge_loaded_weights() {
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
      layers_[i]->merge_loaded_weights();
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  std::pair<int64_t, int64_t> compute_sequence_lengths(
      int64_t seq_length,
      bool is_prefill,
      const ModelInputParams& input_params) const {
    int64_t query_length = seq_length;
    int64_t key_length = seq_length;

    const auto* onerec_params = input_params.onerec_params();
    CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

    if (is_decoder_) {
      if (is_prefill) {
        query_length = seq_length;
        key_length = seq_length;
      } else {
        query_length = 1;
        if (!input_params.kv_seq_lens_vec.empty()) {
          key_length = *std::max_element(input_params.kv_seq_lens_vec.begin(),
                                         input_params.kv_seq_lens_vec.end());
        }
        // Decode keeps a square bias/mask shape expected by OneRec NPU block.
        query_length = key_length;
      }
    } else {
      query_length = onerec_params->encoder_max_seq_len;
      key_length = onerec_params->encoder_max_seq_len;
    }

    return {query_length, key_length};
  }

  torch::Tensor create_moe_attention_mask(int64_t seq_length,
                                          const torch::Tensor& h,
                                          bool is_decoder) const {
    if (!is_decoder) {
      return torch::ones({num_heads_, seq_length, seq_length}, h.options());
    }

    auto mask_value = -9984.0f;
    auto upper_tri_mask =
        torch::triu(torch::ones({seq_length, seq_length},
                                torch::dtype(h.dtype()).device(h.device())),
                    1);
    auto expanded_mask = upper_tri_mask.unsqueeze(0).expand(
        {num_heads_, seq_length, seq_length});
    auto effective_attn_mask =
        torch::zeros({num_heads_, seq_length, seq_length},
                     torch::dtype(h.dtype()).device(h.device()));
    effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool),
                                     mask_value);
    return effective_attn_mask;
  }

  torch::Tensor compute_position_bias_mask(
      int64_t query_length,
      int64_t key_length,
      const torch::Tensor& h,
      bool is_decode_stage,
      const ModelInputParams& input_params) {
    CHECK(!position_bias_embedding_.is_empty())
        << "position_bias_embedding is required for relative attention.";

    auto layer_position_bias =
        compute_onerec_position_bias(query_length,
                                     key_length,
                                     num_heads_,
                                     is_decoder_,
                                     position_bias_embedding_,
                                     relative_attention_num_buckets_,
                                     relative_attention_max_distance_,
                                     torch::dtype(h.dtype()).device(h.device()),
                                     is_decode_stage,
                                     &input_params);

    auto effective_attn_mask = layer_position_bias.is_contiguous()
                                   ? layer_position_bias
                                   : layer_position_bias.contiguous();

    if (is_decoder_ && FLAGS_enable_rec_prefill_only) {
      auto mask_value = -9984.0f;
      auto upper_tri_mask =
          torch::triu(torch::ones({query_length, query_length},
                                  effective_attn_mask.options()),
                      1);
      auto expanded_mask = upper_tri_mask.unsqueeze(0).expand(
          {num_heads_, query_length, query_length});
      effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool),
                                       mask_value);
    }

    return effective_attn_mask;
  }

  torch::Tensor preprocess_attention_mask(
      const torch::Tensor& effective_attn_mask,
      const torch::Tensor& h) const {
    if (!effective_attn_mask.defined()) {
      return torch::Tensor();
    }

    if (effective_attn_mask.device() != h.device()) {
      return effective_attn_mask.to(h.device()).contiguous();
    }
    return effective_attn_mask.is_contiguous()
               ? effective_attn_mask
               : effective_attn_mask.contiguous();
  }

  int64_t hidden_size_ = 0;
  bool is_decoder_ = true;
  bool use_absolute_position_embedding_ = false;
  bool use_moe_ = false;
  int64_t relative_attention_num_buckets_ = 32;
  int64_t relative_attention_max_distance_ = 128;
  int64_t num_heads_ = 4;
  int32_t num_experts_per_tok_ = 2;

  layer::WordEmbedding embed_tokens_{nullptr};
  layer::WordEmbedding position_bias_embedding_{nullptr};
  layer::RMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::NpuOneRecBlockLayer> layers_;
};
TORCH_MODULE(OneRecStack);
#endif

class OneRecModelImpl : public torch::nn::Module {
 public:
  explicit OneRecModelImpl(const ModelContext& context) {
    hidden_size_ = context.get_model_args().hidden_size();
    options_ = context.get_tensor_options();
    shared_ = register_module("shared", layer::WordEmbedding(context));

#if defined(USE_NPU)
    encoder_ = register_module(
        "encoder", OneRecStack(context, /*is_decode=*/false, shared_));
    decoder_ = register_module(
        "decoder", OneRecStack(context, /*is_decode=*/true, shared_));
#endif
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    if (!tokens.defined()) {
      return ModelOutput();
    }
    (void)positions;
    (void)kv_caches;

    const auto* onerec_params = input_params.onerec_params();

#if defined(USE_NPU)
    if (onerec_params != nullptr) {
      if (onerec_params->is_encoder_forward) {
        std::vector<KVCache> encoder_kv_caches;
        auto encoder_output =
            encoder_(tokens, positions, encoder_kv_caches, input_params);

        torch::Tensor cached_encoder_output;
        if (encoder_output.defined() &&
            onerec_params->encoder_max_seq_len > 0 &&
            !onerec_params->encoder_seq_lens.empty()) {
          cached_encoder_output =
              pad_encoder_output(encoder_output, input_params);
        } else {
          cached_encoder_output = encoder_output;
        }
        {
          std::lock_guard<std::mutex> lock(encoder_output_mutex_);
          encoder_output_ = cached_encoder_output;
        }
        return ModelOutput(cached_encoder_output);
      }

      torch::Tensor cached_encoder_output;
      if (onerec_params->has_encoder_output) {
        std::lock_guard<std::mutex> lock(encoder_output_mutex_);
        cached_encoder_output = encoder_output_;
      }

      const torch::Tensor& decoder_context =
          onerec_params->decoder_context_embedding;

      if (!decoder_context.defined() && !cached_encoder_output.defined()) {
        LOG(ERROR)
            << "OneRec decoder requires decoder_context_embedding or encoder "
               "output.";
        return ModelOutput();
      }

      auto decoder_output =
          decoder_(tokens, positions, kv_caches, input_params, cached_encoder_output);
      return ModelOutput(decoder_output);
    }
#endif

    const bool is_encoder_forward =
        (onerec_params != nullptr) && onerec_params->is_encoder_forward;

    auto hidden_states =
        build_hidden_states(tokens, onerec_params, is_encoder_forward);
    if (!hidden_states.defined()) {
      return ModelOutput();
    }

    if (is_encoder_forward) {
      return ModelOutput(hidden_states);
    }

    auto cross_context = resolve_cross_context(onerec_params);
    if (cross_context.defined()) {
      auto enriched_hidden_states =
          add_cross_context_bias(hidden_states, cross_context);
      if (enriched_hidden_states.defined()) {
        hidden_states = std::move(enriched_hidden_states);
      }
    }

    return ModelOutput(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    auto shared_dict = state_dict.get_dict_with_prefix("shared.");
    if (shared_dict.size() > 0) {
      shared_->load_state_dict(shared_dict);
    }

#if defined(USE_NPU)
    auto encoder_dict = state_dict.get_dict_with_prefix("encoder.");
    if (encoder_dict.size() > 0) {
      encoder_->load_state_dict(encoder_dict);
    }
    auto decoder_dict = state_dict.get_dict_with_prefix("decoder.");
    if (decoder_dict.size() > 0) {
      decoder_->load_state_dict(decoder_dict);
    }
#endif
  }

#if defined(USE_NPU)
  void verify_loaded_weights() const {
    encoder_->verify_loaded_weights("encoder.");
    decoder_->verify_loaded_weights("decoder.");
  }

  void merge_loaded_weights() {
    encoder_->merge_loaded_weights();
    decoder_->merge_loaded_weights();
  }
#endif

  layer::WordEmbedding get_word_embedding() { return shared_; }

  void set_word_embedding(layer::WordEmbedding& embedding) {
    shared_ = embedding;
#if defined(USE_NPU)
    encoder_->set_word_embedding(shared_);
    decoder_->set_word_embedding(shared_);
#endif
  }

 private:
  static bool is_token_id_tensor(const torch::Tensor& tokens) {
    return tokens.scalar_type() == torch::kLong ||
           tokens.scalar_type() == torch::kInt;
  }

  torch::Tensor build_hidden_states(const torch::Tensor& tokens,
                                    const OneRecModelInputParams* onerec_params,
                                    bool is_encoder_forward) {
    if (tokens.numel() == 0) {
      return torch::empty({0, hidden_size_}, options_);
    }

    if (is_token_id_tensor(tokens)) {
      return shared_(tokens);
    }

    if (tokens.dim() == 2 && tokens.size(-1) == hidden_size_) {
      if (onerec_params != nullptr) {
        if (onerec_params->is_hybrid_mode || is_encoder_forward) {
          return tokens;
        }
        if (onerec_params->decoder_context_embedding.defined()) {
          return tokens;
        }
      }
      return tokens;
    }

    LOG(ERROR) << "Invalid OneRec token tensor shape for non-id path: "
               << tokens.sizes();
    return torch::Tensor();
  }

  torch::Tensor resolve_cross_context(
      const OneRecModelInputParams* onerec_params) const {
    if (onerec_params == nullptr) {
      return torch::Tensor();
    }
    if (onerec_params->decoder_context_embedding.defined()) {
      return onerec_params->decoder_context_embedding;
    }
    return torch::Tensor();
  }

  torch::Tensor add_cross_context_bias(
      const torch::Tensor& hidden_states,
      const torch::Tensor& cross_context) const {
    if (!hidden_states.defined() || !cross_context.defined()) {
      return hidden_states;
    }

    if (hidden_states.dim() != 2 || hidden_states.size(-1) != hidden_size_) {
      LOG(ERROR) << "Unexpected hidden_states shape in OneRec decoder: "
                 << hidden_states.sizes();
      return hidden_states;
    }

    auto context = cross_context;
    if (context.device() != hidden_states.device()) {
      context = context.to(hidden_states.device());
    }
    if (context.dtype() != hidden_states.dtype()) {
      context = context.to(hidden_states.dtype());
    }

    if (context.dim() == 1 && context.size(0) == hidden_size_) {
      context = context.unsqueeze(0);
    } else if (context.dim() > 2 && context.size(-1) == hidden_size_) {
      context = context.reshape({-1, hidden_size_});
    }

    if (context.dim() != 2 || context.size(-1) != hidden_size_) {
      LOG(ERROR) << "Unexpected OneRec cross context shape: "
                 << context.sizes();
      return hidden_states;
    }

    auto pooled_context = context.mean(/*dim=*/0, /*keepdim=*/true);
    return hidden_states + pooled_context.expand(
                               {hidden_states.size(0), pooled_context.size(1)});
  }

  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  layer::WordEmbedding shared_{nullptr};

#if defined(USE_NPU)
  OneRecStack encoder_{nullptr};
  OneRecStack decoder_{nullptr};
  torch::Tensor encoder_output_;
  std::mutex encoder_output_mutex_;
#endif
};
TORCH_MODULE(OneRecModel);

class OneRecForConditionalGenerationImpl
    : public RecForCausalLMImplBase<OneRecModel> {
 public:
  explicit OneRecForConditionalGenerationImpl(const ModelContext& context)
      : RecForCausalLMImplBase<OneRecModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      StateDict model_state_dict = state_dict->get_dict_with_prefix(prefix);
      if (model_state_dict.size() == 0) {
        model_state_dict = *state_dict;
      }
      model_->load_state_dict(model_state_dict);

      if (tie_word_embeddings_) {
        auto shared_dict = model_state_dict.get_dict_with_prefix("shared.");
        if (shared_dict.size() == 0) {
          shared_dict = state_dict->get_dict_with_prefix("shared.");
        }
        if (shared_dict.size() != 0) {
          lm_head_->load_state_dict(shared_dict);
        }
      } else {
        auto lm_head_dict = model_state_dict.get_dict_with_prefix("lm_head.");
        if (lm_head_dict.size() == 0) {
          lm_head_dict = state_dict->get_dict_with_prefix("lm_head.");
        }
        if (lm_head_dict.size() != 0) {
          lm_head_->load_state_dict(lm_head_dict);
        }
      }
    }

#if defined(USE_NPU)
    model_->verify_loaded_weights();
    model_->merge_loaded_weights();
#endif
  }
};
TORCH_MODULE(OneRecForConditionalGeneration);

using OneRecCausalLM = CausalLMImpl<OneRecForConditionalGeneration>;
static_assert(std::is_base_of_v<CausalLM, OneRecCausalLM>,
              "OneRec must satisfy CausalLM contract.");

REGISTER_REC_MODEL(onerec, OneRecForConditionalGeneration);

REGISTER_MODEL_ARGS(onerec, [&] {
  LOAD_ARG_OR(model_type, "model_type", "onerec");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");

  LOAD_ARG_OR(hidden_size, "d_model", 128);
  LOAD_ARG_OR(intermediate_size, "d_ff", 256);

  LOAD_ARG_OR(n_layers, "num_decoder_layers", 4);
  LOAD_ARG_OR(n_encoder_layers, "num_layers", 12);

  LOAD_ARG_OR(n_heads, "num_heads", 4);
  LOAD_ARG_OR(head_dim, "d_kv", 32);
  LOAD_ARG_OR_FUNC(
      decoder_n_heads, "decoder_num_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      decoder_head_dim, "decoder_d_kv", [&] { return args->head_dim(); });

  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG(decoder_n_kv_heads, "decoder_num_key_value_heads");

  LOAD_ARG_OR(vocab_size, "vocab_size", 8200);
  LOAD_ARG_OR(rms_norm_eps, "layer_norm_epsilon", 1e-6);
  LOAD_ARG_OR(max_position_embeddings, "max_length", 500);
  LOAD_ARG_OR(use_absolute_position_embedding,
              "use_absolute_position_embedding",
              false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", true);

  LOAD_ARG_OR(use_moe, "use_moe", false);
  LOAD_ARG_OR(moe_score_func, "moe_score_func", "softmax");
  LOAD_ARG_OR(moe_route_scale, "moe_route_scale", 1.0f);
  LOAD_ARG_OR(n_routed_experts, "moe_num_experts", 8);
  LOAD_ARG_OR(moe_use_shared_experts, "moe_use_shared_experts", false);
  LOAD_ARG_OR(n_shared_experts, "moe_num_shared_experts", 0);
  LOAD_ARG_OR(num_experts_per_tok, "moe_topk", 2);
  LOAD_ARG_OR(moe_intermediate_size, "moe_inter_dim", 1024);

  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

REGISTER_TOKENIZER_ARGS(onerec, [&] {
  SET_ARG(tokenizer_type, "rec");
  LOAD_ARG_OR(vocab_file, "vocab_file", "");
});

}  // namespace xllm
