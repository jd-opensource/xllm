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

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/state_dict/utils.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "layers/npu/deepseek_v4_rotary_embedding.h"
#include "models/llm/deepseek_v4.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mtp_model_base.h"

namespace xllm {

class DeepseekV4MultiTokenPredictorLayerImpl
    : public MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  DeepseekV4MultiTokenPredictorLayerImpl(const ModelContext& context,
                                         int32_t layer_index)
      : MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>(context,
                                                               layer_index) {}

  torch::Tensor forward(torch::Tensor inputs_embeds,
                        torch::Tensor previous_hidden_states,
                        torch::Tensor positions,
                        layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor tokens) {
    ModelInputParams modified_input_params = input_params;
    modified_input_params.input_embedding = previous_hidden_states;
    std::optional<torch::Tensor> residual;
    return MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>::forward(
        inputs_embeds,
        residual,
        positions,
        attn_metadata,
        kv_cache,
        modified_input_params,
        tokens);
  }
};
TORCH_MODULE(DeepseekV4MultiTokenPredictorLayer);

class DeepseekV4MtpModelImpl : public torch::nn::Module {
 public:
  explicit DeepseekV4MtpModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    model_args_ = &model_args;

    CHECK_GT(model_args.n_layers(), 0)
        << "deepseek_v4_mtp requires n_layers > 0";
    CHECK_GE(model_args.num_nextn_predict_layers(), 0)
        << "deepseek_v4_mtp requires num_nextn_predict_layers >= 0";

    const int32_t mtp_n_layers = model_args.n_layers();

    num_heads_ = model_args.n_heads();
    head_dim_ = model_args.o_lora_rank() + model_args.qk_rope_head_dim();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DeepseekV4Mtp] n_heads must be divisible by local tp "
           "size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_;
    tp_num_heads_ = num_heads_ / dp_local_tp_size_;
    window_size_ = model_args.window_size();
    index_n_heads_ = model_args.index_n_heads();
    index_head_dim_ = model_args.index_head_dim();
    index_topk_ = model_args.index_topk();
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim > 0 && max_pos > 0) {
      const int64_t original_max_pos =
          model_args.rope_scaling_original_max_position_embeddings() > 0
              ? model_args.rope_scaling_original_max_position_embeddings()
              : max_pos;
      dsa_rotary_embedding_ =
          std::make_shared<layer::DeepseekV4RotaryEmbedding>(
              /*rotary_dim=*/rope_head_dim,
              /*max_position_embeddings=*/max_pos,
              /*interleaved=*/true,
              /*rope_theta=*/model_args.rope_theta(),
              /*compress_rope_theta=*/model_args.compress_rope_theta(),
              /*scaling_factor=*/model_args.factor(),
              /*extrapolation_factor=*/1.0f,
              /*beta_fast=*/model_args.beta_fast(),
              /*beta_slow=*/model_args.beta_slow(),
              /*attn_factor=*/model_args.rope_scaling_attn_factor(),
              /*mscale=*/1.0f,
              /*mscale_all_dim=*/1.0f,
              /*original_max_position_embeddings=*/original_max_pos,
              options);
      dsa_cos_sin_ = dsa_rotary_embedding_->get_cos_sin_cache("default");
    }

    if (model_args.index_head_dim() > 0) {
      auto hadamard_dim_padded =
          deepseek_v4_next_power_of_two(model_args.index_head_dim());
      dsa_hadamard_ = deepseek_v4_create_hadamard_matrix(
          hadamard_dim_padded, options.dtype().toScalarType(), options.device());
    }

    deepseek_v4_build_cache_specs(model_args, caches_info_, group_infos_);

    mtp_layers_.reserve(mtp_n_layers);
    for (int32_t i = 0; i < mtp_n_layers; ++i) {
      const int32_t layer_index = i;
      mtp_layers_.push_back(
          DeepseekV4MultiTokenPredictorLayer(context, layer_index));
      register_module("layer_" + std::to_string(i), mtp_layers_.back());
    }

    final_norm_ = register_module("final_norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      mtp_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(i) + "."));
    }

    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.norm."));
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.emb.tok_emb."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    UNUSED_PARAMETER(prefix);
    for (const auto& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
  }

  void merge_loaded_weights() {
    for (const auto& layer : mtp_layers_) {
      UNUSED_PARAMETER(layer);
    }
  }

  void merge_and_move_pinned_host() {
    merge_loaded_weights();
  }

  void free_weights() {}

  void reload_weights() {}

  void reload_non_decoder_weights() {}

  void reload_weights_from_device() {}

  void refresh_rolling_weights() {}

  layer::WordEmbedding get_word_embedding() {
    return embed_tokens_;
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }

    const torch::Device runtime_device = tokens.device();

    torch::Tensor previous_hidden_states = input_params.input_embedding;
    CHECK(previous_hidden_states.defined())
        << "input_params.input_embedding must be defined for MTP model";

    torch::Tensor hidden_states = embed_tokens_(tokens);

    const bool acl_graph_forward = deepseek_v4_uses_acl_graph(input_params);
    if (acl_graph_forward) {
      CHECK(tokens.defined() && tokens.device() == runtime_device)
          << "[DeepseekV4Mtp] ACL graph requires tokens on the runtime device";
      CHECK(positions.defined() && positions.device() == runtime_device)
          << "[DeepseekV4Mtp] ACL graph requires positions on the runtime device";
      CHECK(input_params.new_cache_slots.defined())
          << "[DeepseekV4Mtp] ACL graph requires persistent new_cache_slots";
      CHECK(input_params.block_tables.defined())
          << "[DeepseekV4Mtp] ACL graph requires persistent block_tables";
    } else {
      tokens = maybe_to_device(tokens, runtime_device);
      positions = maybe_to_device(positions, runtime_device);
    }

    auto mask = (positions == 0);
    if (mask.any().item<bool>()) {
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      CHECK(!acl_graph_forward)
          << "[DeepseekV4Mtp] ACL graph requires prebuilt attention metadata";
      modified_input_params.attn_metadata =
          build_attention_metadata_for_forward(positions, modified_input_params);
    }

    CHECK_GE(static_cast<int32_t>(kv_caches.size()),
             static_cast<int32_t>(mtp_layers_.size()))
        << "deepseek_v4_mtp requires kv_caches size >= mtp layer count";

    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      const int32_t layer_id = static_cast<int32_t>(i);
      prepare_for_layer(*modified_input_params.attn_metadata, layer_id);
      hidden_states = mtp_layers_[i](hidden_states,
                                     previous_hidden_states,
                                     positions,
                                     *modified_input_params.attn_metadata,
                                     kv_caches[i],
                                     modified_input_params,
                                     tokens);
    }

    auto [output, _] = final_norm_(hidden_states, std::nullopt);
    return ModelOutput(output, std::nullopt);
  }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<DeepseekV4GraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    CHECK(state != nullptr)
        << "[DeepseekV4Mtp] graph metadata state must be initialized";
    auto* mtp_state = dynamic_cast<DeepseekV4GraphMetadataState*>(state);
    CHECK(mtp_state != nullptr)
        << "[DeepseekV4Mtp] received incompatible graph metadata state";
    DeepseekV4ModelImpl::normalize_graph_padding(input_params);

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilder::build(modified_input_params,
                                         positions,
                                         dsa_cos_sin_,
                                         caches_info_,
                                         group_infos_));
    if (attn_metadata->dsa_metadata) {
      auto& dsa = *attn_metadata->dsa_metadata;
      if (dsa_hadamard_.defined()) {
        dsa.hadamard = dsa_hadamard_;
      }
      DeepseekV4ModelImpl::copy_to_graph_packed_metadata_buffer(
          dsa, mtp_state->dsa_metadata_persistent, positions.device());
      prepare_for_forward(
          *attn_metadata, positions.device(), /*pack_metadata=*/false);
    }
    input_params.attn_metadata = persist_graph_attention_metadata(
        *mtp_state, std::move(attn_metadata));
    CHECK(input_params.attn_metadata)
        << "[DeepseekV4Mtp] ACL graph requires DSA metadata";
  }

 private:
  std::shared_ptr<layer::AttentionMetadata>
  build_attention_metadata_for_forward(const torch::Tensor& positions,
                                       const ModelInputParams& input_params) {
    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilder::build(modified_input_params,
                                         positions,
                                         dsa_cos_sin_,
                                         caches_info_,
                                         group_infos_));
    if (attn_metadata->dsa_metadata) {
      prepare_for_forward(*attn_metadata, positions.device());
    }
    return attn_metadata;
  }

  void prepare_for_forward(layer::AttentionMetadata& attn_metadata,
                           const torch::Device& runtime_device,
                           bool pack_metadata = true) const {
    CHECK(attn_metadata.dsa_metadata)
        << "[DeepseekV4Mtp] attn_metadata.dsa_metadata must be populated";

    auto& dsa = *(attn_metadata.dsa_metadata);

    dsa.seq_lens = maybe_to_device(dsa.seq_lens, runtime_device);
    dsa.seq_lens_q =
        maybe_to_device(dsa.seq_lens_q, runtime_device);
    dsa.actual_seq_lengths_query =
        maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
    dsa.actual_seq_lengths_kv =
        maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
    dsa.max_seqlen_q =
        maybe_to_device(dsa.max_seqlen_q, runtime_device);
    dsa.max_seqlen_kv =
        maybe_to_device(dsa.max_seqlen_kv, runtime_device);
    dsa.input_positions =
        maybe_to_device(dsa.input_positions, runtime_device);
    dsa.c4_pad_positions =
        maybe_to_device(dsa.c4_pad_positions, runtime_device);
    dsa.c128_pad_positions =
        maybe_to_device(dsa.c128_pad_positions, runtime_device);

    for (auto& layer_block_tables : dsa.block_tables) {
      for (auto& block_table : layer_block_tables) {
        block_table = maybe_to_device(block_table, runtime_device);
      }
    }
    for (auto& layer_slot_mappings : dsa.slot_mappings) {
      for (auto& slot_mapping : layer_slot_mappings) {
        slot_mapping = maybe_to_device(slot_mapping, runtime_device);
      }
    }

    if (dsa_hadamard_.defined()) {
      dsa.hadamard = maybe_to_device(dsa_hadamard_, runtime_device);
    }
    if (pack_metadata) {
      deepseek_v4_pack_dsa_metadata_to_device(dsa, runtime_device);
    }

    build_rope(dsa, runtime_device);

    if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
      dsa.start_pos =
          (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
    }

    build_precomputed_metadata(dsa);
  }

  void build_rope(layer::DSAMetadata& dsa,
                  const torch::Device& runtime_device) const {
    if (!dsa_rotary_embedding_) {
      return;
    }

    std::unordered_map<std::string, torch::Tensor> positions_map;
    dsa.cos = torch::Tensor();
    dsa.sin = torch::Tensor();
    dsa.c4_cos = torch::Tensor();
    dsa.c4_sin = torch::Tensor();
    dsa.c128_cos = torch::Tensor();
    dsa.c128_sin = torch::Tensor();
    dsa.c4_input_cos = torch::Tensor();
    dsa.c4_input_sin = torch::Tensor();
    dsa.c128_input_cos = torch::Tensor();
    dsa.c128_input_sin = torch::Tensor();

    auto append_group_positions = [&positions_map](const std::string& group,
                                                   const torch::Tensor& positions) {
      if (!positions.defined() || positions.numel() == 0) {
        return;
      }
      auto group_positions = positions;
      if (group_positions.scalar_type() != torch::kInt64) {
        group_positions = group_positions.to(torch::kInt64);
      }
      positions_map[group] = group_positions;
    };

    append_group_positions("default", dsa.input_positions);
    append_group_positions("c4", dsa.c4_pad_positions);
    append_group_positions("c128", dsa.c128_pad_positions);

    if (!positions_map.empty()) {
      auto group_cos_sin = dsa_rotary_embedding_->build(positions_map);

      auto default_it = group_cos_sin.find("default");
      if (default_it != group_cos_sin.end()) {
        dsa.cos = default_it->second.first;
        dsa.sin = default_it->second.second;
      }

      auto c4_it = group_cos_sin.find("c4");
      if (c4_it != group_cos_sin.end()) {
        dsa.c4_cos = c4_it->second.first;
        dsa.c4_sin = c4_it->second.second;
      }

      auto c128_it = group_cos_sin.find("c128");
      if (c128_it != group_cos_sin.end()) {
        dsa.c128_cos = c128_it->second.first;
        dsa.c128_sin = c128_it->second.second;
      }
    }

    dsa.cos = maybe_to_device(dsa.cos, runtime_device);
    dsa.sin = maybe_to_device(dsa.sin, runtime_device);
    dsa.c4_cos = maybe_to_device(dsa.c4_cos, runtime_device);
    dsa.c4_sin = maybe_to_device(dsa.c4_sin, runtime_device);
    dsa.c128_cos =
        maybe_to_device(dsa.c128_cos, runtime_device);
    dsa.c128_sin =
        maybe_to_device(dsa.c128_sin, runtime_device);

    if (dsa.input_positions.defined() && dsa.input_positions.numel() > 0) {
      auto input_positions = dsa.input_positions;
      if (input_positions.scalar_type() != torch::kInt64) {
        input_positions = input_positions.to(torch::kInt64);
      }
      auto input_group_cos_sin = dsa_rotary_embedding_->build(
          {{"c4", input_positions}, {"c128", input_positions}});
      auto c4_input_it = input_group_cos_sin.find("c4");
      if (c4_input_it != input_group_cos_sin.end()) {
        dsa.c4_input_cos = c4_input_it->second.first;
        dsa.c4_input_sin = c4_input_it->second.second;
      }
      auto c128_input_it = input_group_cos_sin.find("c128");
      if (c128_input_it != input_group_cos_sin.end()) {
        dsa.c128_input_cos = c128_input_it->second.first;
        dsa.c128_input_sin = c128_input_it->second.second;
      }
    }
  }

  void build_precomputed_metadata(layer::DSAMetadata& dsa) const {
    dsa.c1_metadata = torch::Tensor();
    dsa.c4_metadata = torch::Tensor();
    dsa.c128_metadata = torch::Tensor();
    dsa.qli_metadata = torch::Tensor();

    torch::Device metadata_device(torch::kCPU);
    if (dsa.input_positions.defined()) {
      metadata_device = dsa.input_positions.device();
    } else if (dsa.seq_lens_q.defined()) {
      metadata_device = dsa.seq_lens_q.device();
    } else if (dsa.actual_seq_lengths_kv.defined()) {
      metadata_device = dsa.actual_seq_lengths_kv.device();
    }

    dsa.actual_seq_lengths_query =
        maybe_to_device(dsa.actual_seq_lengths_query, metadata_device);
    dsa.actual_seq_lengths_kv =
        maybe_to_device(dsa.actual_seq_lengths_kv, metadata_device);
    dsa.seq_lens_q =
        maybe_to_device(dsa.seq_lens_q, metadata_device);
    dsa.seq_lens =
        maybe_to_device(dsa.seq_lens, metadata_device);
    dsa.max_seqlen_q =
        maybe_to_device(dsa.max_seqlen_q, metadata_device);
    dsa.max_seqlen_kv =
        maybe_to_device(dsa.max_seqlen_kv, metadata_device);

    if (!dsa.actual_seq_lengths_query.defined() ||
        !dsa.actual_seq_lengths_kv.defined()) {
      return;
    }

    const int64_t batch_size =
        std::max<int64_t>(dsa.actual_seq_lengths_kv.size(0), 1);
    const int64_t max_seqlen_q =
        dsa.max_seqlen_q.defined() && dsa.max_seqlen_q.numel() > 0
            ? dsa.max_seqlen_q.max().item<int64_t>()
            : (dsa.seq_lens_q.defined() && dsa.seq_lens_q.numel() > 0
                   ? dsa.seq_lens_q.max().item<int64_t>()
                   : 0);
    const int64_t max_seqlen_kv =
        dsa.max_seqlen_kv.defined() && dsa.max_seqlen_kv.numel() > 0
            ? dsa.max_seqlen_kv.max().item<int64_t>()
            : (dsa.actual_seq_lengths_kv.defined() &&
                       dsa.actual_seq_lengths_kv.numel() > 0
                   ? dsa.actual_seq_lengths_kv.max().item<int64_t>()
                   : 0);
    const int64_t ori_win_left = std::max<int64_t>(window_size_ - 1, 0);
    const int64_t sparse_topk = std::max<int64_t>(index_topk_, 1);
    const bool is_prefill =
        dsa.seq_lens_q.defined() && dsa.seq_lens_q.numel() > 0 &&
        dsa.seq_lens_q.max().item<int64_t>() > 1;

    const char* layout_kv = "PA_ND";
    auto cu_seqlens_ori_kv_opt =
        is_prefill ? (dsa.actual_seq_lengths_query.defined() &&
                             dsa.actual_seq_lengths_query.numel() > 0
                         ? c10::optional<torch::Tensor>(dsa.actual_seq_lengths_query)
                         : c10::nullopt)
                   : c10::nullopt;

    auto as_optional_tensor = [](const torch::Tensor& tensor)
        -> c10::optional<torch::Tensor> {
      if (tensor.defined() && tensor.numel() > 0) {
        return c10::optional<torch::Tensor>(tensor);
      }
      return c10::nullopt;
    };

    xllm::kernel::SparseAttnSharedkvMetadataParams c1_params;
    c1_params.num_heads_q = tp_num_heads_;
    c1_params.num_heads_kv = 1;
    c1_params.head_dim = head_dim_;
    c1_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c1_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c1_params.cu_seqlens_cmp_kv = c10::nullopt;
    c1_params.seqused_q = c10::nullopt;
    c1_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c1_params.batch_size = batch_size;
    c1_params.max_seqlen_q = max_seqlen_q;
    c1_params.max_seqlen_kv = max_seqlen_kv;
    c1_params.ori_topk = 0;
    c1_params.cmp_topk = 0;
    c1_params.cmp_ratio = 1;
    c1_params.ori_mask_mode = 4;
    c1_params.cmp_mask_mode = 3;
    c1_params.ori_win_left = ori_win_left;
    c1_params.ori_win_right = 0;
    c1_params.layout_q = "TND";
    c1_params.layout_kv = layout_kv;
    c1_params.has_ori_kv = true;
    c1_params.has_cmp_kv = false;
    dsa.c1_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c1_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c4_params;
    c4_params.num_heads_q = tp_num_heads_;
    c4_params.num_heads_kv = 1;
    c4_params.head_dim = head_dim_;
    c4_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c4_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c4_params.cu_seqlens_cmp_kv = c10::nullopt;
    c4_params.seqused_q = c10::nullopt;
    c4_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c4_params.batch_size = batch_size;
    c4_params.max_seqlen_q = max_seqlen_q;
    c4_params.max_seqlen_kv = max_seqlen_kv;
    c4_params.ori_topk = 0;
    c4_params.cmp_topk = sparse_topk;
    c4_params.cmp_ratio = 4;
    c4_params.ori_mask_mode = 4;
    c4_params.cmp_mask_mode = 3;
    c4_params.ori_win_left = ori_win_left;
    c4_params.ori_win_right = 0;
    c4_params.layout_q = "TND";
    c4_params.layout_kv = layout_kv;
    c4_params.has_ori_kv = true;
    c4_params.has_cmp_kv = true;
    dsa.c4_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c4_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c128_params;
    c128_params.num_heads_q = tp_num_heads_;
    c128_params.num_heads_kv = 1;
    c128_params.head_dim = head_dim_;
    c128_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c128_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c128_params.cu_seqlens_cmp_kv = c10::nullopt;
    c128_params.seqused_q = c10::nullopt;
    c128_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c128_params.batch_size = batch_size;
    c128_params.max_seqlen_q = max_seqlen_q;
    c128_params.max_seqlen_kv = max_seqlen_kv;
    c128_params.ori_topk = 0;
    c128_params.cmp_topk = 0;
    c128_params.cmp_ratio = 128;
    c128_params.ori_mask_mode = 4;
    c128_params.cmp_mask_mode = 3;
    c128_params.ori_win_left = ori_win_left;
    c128_params.ori_win_right = 0;
    c128_params.layout_q = "TND";
    c128_params.layout_kv = layout_kv;
    c128_params.has_ori_kv = true;
    c128_params.has_cmp_kv = true;
    dsa.c128_metadata =
        xllm::kernel::sparse_attn_sharedkv_metadata(c128_params);

    torch::Tensor query_lens;
    if (dsa.actual_seq_lengths_query.defined() &&
        dsa.actual_seq_lengths_query.dim() > 0 &&
        dsa.actual_seq_lengths_query.size(0) > 1) {
      query_lens = dsa.actual_seq_lengths_query
                       .slice(/*dim=*/0,
                              /*start=*/1,
                              /*end=*/dsa.actual_seq_lengths_query.size(0))
                       .clone();
    } else if (dsa.seq_lens_q.defined()) {
      query_lens = dsa.seq_lens_q;
    }

    torch::Tensor key_lens;
    if (dsa.seq_lens.defined()) {
      key_lens = dsa.seq_lens;
    } else if (dsa.actual_seq_lengths_kv.defined()) {
      key_lens = dsa.actual_seq_lengths_kv;
    }

    if (!query_lens.defined() || !key_lens.defined() ||
        query_lens.numel() == 0 || key_lens.numel() == 0) {
      return;
    }

    const int64_t global_index_num_heads =
        std::max<int64_t>(index_n_heads_ > 0 ? index_n_heads_ : num_heads_, 1);
    CHECK_EQ(global_index_num_heads % dp_local_tp_size_, 0)
        << "[DeepseekV4Mtp] index/global heads must be divisible "
           "by local tp size. global_index_num_heads="
        << global_index_num_heads << ", local_tp_size=" << dp_local_tp_size_;
    const int64_t index_num_heads =
        std::max<int64_t>(global_index_num_heads / dp_local_tp_size_, 1);
    const int64_t index_head_dim =
        std::max<int64_t>(index_head_dim_ > 0 ? index_head_dim_ : head_dim_, 1);
    const int64_t qli_batch_size = std::max<int64_t>(key_lens.size(0), 1);
    const int64_t qli_max_seqlen_q =
        dsa.max_seqlen_q.defined() && dsa.max_seqlen_q.numel() > 0
            ? dsa.max_seqlen_q.max().item<int64_t>()
            : (query_lens.defined() && query_lens.numel() > 0
                   ? query_lens.max().item<int64_t>()
                   : 0);
    const int64_t qli_max_seqlen_k =
        dsa.max_seqlen_kv.defined() && dsa.max_seqlen_kv.numel() > 0
            ? dsa.max_seqlen_kv.max().item<int64_t>()
            : (key_lens.defined() && key_lens.numel() > 0
                   ? key_lens.max().item<int64_t>()
                   : 0);

    xllm::kernel::QuantLightningIndexerMetadataParams qli_params;
    qli_params.num_heads_q = global_index_num_heads;
    qli_params.num_heads_k = 1;
    qli_params.head_dim = index_head_dim;
    qli_params.query_quant_mode = 0;
    qli_params.key_quant_mode = 0;
    qli_params.actual_seq_lengths_query = as_optional_tensor(query_lens);
    qli_params.actual_seq_lengths_key = as_optional_tensor(key_lens);
    qli_params.batch_size = qli_batch_size;
    qli_params.max_seqlen_q = qli_max_seqlen_q;
    qli_params.max_seqlen_k = qli_max_seqlen_k;
    qli_params.layout_query = "TND";
    qli_params.layout_key = "PA_BSND";
    qli_params.sparse_count = sparse_topk;
    qli_params.sparse_mode = 3;
    qli_params.pre_tokens = std::numeric_limits<int64_t>::max();
    qli_params.next_tokens = std::numeric_limits<int64_t>::max();
    qli_params.cmp_ratio = 4;
    qli_params.device = query_lens.device().str();
    dsa.qli_metadata =
        xllm::kernel::quant_lightning_indexer_metadata(qli_params);
  }

  void prepare_for_layer(layer::AttentionMetadata& attn_metadata,
                         int32_t layer_id) const {
    CHECK(attn_metadata.dsa_metadata)
        << "[DeepseekV4Mtp] attn_metadata.dsa_metadata must be populated";

    auto& dsa = *(attn_metadata.dsa_metadata);
    dsa.layer_id = layer_id;

    const int32_t layer_compress_ratio =
        deepseek_v4_normalize_compress_ratio(
            (layer_id <
             static_cast<int32_t>(model_args_->compress_ratios().size()))
                ? model_args_->compress_ratios()[static_cast<size_t>(layer_id)]
                : 1);

    if (layer_compress_ratio == 4 && dsa.c4_cos.defined()) {
      dsa.cos = dsa.c4_cos;
      dsa.sin = dsa.c4_sin;
    } else if (layer_compress_ratio == 128 && dsa.c128_cos.defined()) {
      dsa.cos = dsa.c128_cos;
      dsa.sin = dsa.c128_sin;
    }

    if (layer_id < static_cast<int32_t>(dsa.block_tables.size()) &&
        layer_id < static_cast<int32_t>(dsa.slot_mappings.size()) &&
        !dsa.block_tables[layer_id].empty() &&
        !dsa.slot_mappings[layer_id].empty()) {
      size_t attn_cache_idx = 0;
      if (layer_id < static_cast<int32_t>(caches_info_.size())) {
        const auto& layer_caches = caches_info_[layer_id];
        for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
          if (layer_caches[cache_idx].type ==
              DSACacheType::SLIDING_WINDOW) {
            attn_cache_idx = cache_idx;
            break;
          }
        }
      }

      if (attn_cache_idx < dsa.block_tables[layer_id].size() &&
          dsa.block_tables[layer_id][attn_cache_idx].defined()) {
        attn_metadata.block_table =
            dsa.block_tables[layer_id][attn_cache_idx];
      }
      if (attn_cache_idx < dsa.slot_mappings[layer_id].size() &&
          dsa.slot_mappings[layer_id][attn_cache_idx].defined()) {
        attn_metadata.slot_mapping =
            dsa.slot_mappings[layer_id][attn_cache_idx];
      }
    }
  }

  std::shared_ptr<layer::AttentionMetadata> persist_graph_attention_metadata(
      DeepseekV4GraphMetadataState& state,
      std::shared_ptr<layer::AttentionMetadata> metadata) const {
    return DeepseekV4ModelImpl::persist_graph_attention_metadata(
        state, std::move(metadata));
  }

  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;
  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_hadamard_;

  std::vector<std::vector<DSACacheInfo>> caches_info_;
  std::vector<DSAGroupInfo> group_infos_;

  int64_t num_heads_ = 0;
  int64_t tp_num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
  int64_t head_dim_ = 0;
  int64_t window_size_ = 128;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 512;
  double norm_eps_ = 1e-6;

  const ModelArgs* model_args_ = nullptr;

  layer::RMSNorm final_norm_{nullptr};
  layer::WordEmbedding embed_tokens_{nullptr};
  std::vector<DeepseekV4MultiTokenPredictorLayer> mtp_layers_;
};
TORCH_MODULE(DeepseekV4MtpModel);

class DeepseekV4MtpForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4MtpModel> {
 public:
  explicit DeepseekV4MtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4MtpModel>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      lm_head_->load_state_dict(
          state_dict->get_dict_with_prefix(prefix + "layers.0.head."));
    }
    model_->verify_loaded_weights(prefix);
  }

  bool requires_graph_forward_metadata() {
    return this->model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return this->model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    this->model_->prepare_graph_forward_metadata(
        state, positions, input_params);
  }
};
TORCH_MODULE(DeepseekV4MtpForCausalLM);

inline void load_deepseek_v4_mtp_model_args(const JsonReader& json,
                                            ModelArgs* args) {
  load_deepseek_v4_model_args(json, args);
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4_mtp");
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
  SET_ARG(n_hash_layers, 0);
}

REGISTER_CAUSAL_MODEL(deepseek_v4_mtp, DeepseekV4MtpForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4_mtp, [&] {
  constexpr auto preset = DeepseekV4PolicyPreset::kDefault;
  const auto args_policy = build_deepseek_v4_args_policy(preset);
  load_deepseek_v4_mtp_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
  normalize_deepseek_v4_args(args);
});

}  // namespace xllm