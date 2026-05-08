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

namespace xllm {

inline int64_t deepseek_v4_mtp_next_power_of_two(int64_t n) {
  int64_t value = 1;
  while (value < n) {
    value <<= 1;
  }
  return value;
}

inline torch::Tensor deepseek_v4_mtp_create_hadamard_matrix(
    int64_t n,
    torch::ScalarType dtype,
    const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    auto top = torch::cat({matrix, matrix}, 1);
    auto bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  return matrix;
}

inline torch::Tensor deepseek_v4_mtp_maybe_to_device(const torch::Tensor& tensor,
                                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline int32_t deepseek_v4_mtp_normalize_compress_ratio(int32_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

class DeepseekV4MultiTokenPredictorLayerImpl : public torch::nn::Module {
 public:
  DeepseekV4MultiTokenPredictorLayerImpl(const ModelContext& context,
                                         int32_t layer_index)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();
    enorm_ = register_module("enorm", layer::RMSNorm(context));
    hnorm_ = register_module("hnorm", layer::RMSNorm(context));
    e_proj_ = register_module(
        "e_proj",
        layer::ReplicatedLinear(model_args_.hidden_size(),
                                model_args_.hidden_size(),
                                false,
                                QuantArgs(),
                                options));
    h_proj_ = register_module(
        "h_proj",
        layer::ReplicatedLinear(model_args_.hidden_size(),
                                model_args_.hidden_size(),
                                false,
                                QuantArgs(),
                                options));

    const int32_t runtime_layer_index =
        std::min<int32_t>(layer_index, model_args_.n_layers() - 1);
    mtp_block_ = register_module(
        "mtp_block", layer::DeepseekV4DecoderLayer(context, runtime_layer_index));

    hc_mult_ = model_args_.hc_mult();
    hc_eps_ = static_cast<double>(model_args_.hc_eps());
    norm_eps_ = static_cast<double>(model_args_.rms_norm_eps());
    const int64_t hc_dim = hc_mult_ * model_args_.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ = register_parameter(
        "hc_head_fn", torch::empty({hc_mult_, hc_dim}, hc_options), false);
    hc_head_base_ = register_parameter(
        "hc_head_base", torch::empty({hc_mult_}, hc_options), false);
    hc_head_scale_ = register_parameter(
        "hc_head_scale", torch::empty({1}, hc_options), false);
  }

  void load_state_dict(const StateDict& state_dict) {
    e_proj_->load_state_dict(state_dict.get_dict_with_prefix("e_proj."));
    h_proj_->load_state_dict(state_dict.get_dict_with_prefix("h_proj."));
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    mtp_block_->load_state_dict(state_dict);
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  void verify_loaded_weights() const {}

  torch::Tensor forward(torch::Tensor inputs_embeds,
                        torch::Tensor previous_hidden_states,
                        torch::Tensor positions,
                        layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor tokens) {
    torch::NoGradGuard no_grad;

    auto [e_norm, _1] = enorm_(inputs_embeds, std::nullopt);
    auto [h_norm, _2] = hnorm_(previous_hidden_states, std::nullopt);
    auto hidden_states = e_proj_(e_norm) + h_proj_(h_norm);

    if (hidden_states.dim() == 2) {
      hidden_states = hidden_states.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    auto residual = c10::optional<torch::Tensor>();
    hidden_states = mtp_block_(hidden_states,
                               residual,
                               positions,
                               attn_metadata,
                               kv_cache,
                               input_params,
                               tokens);

    return hc_head(hidden_states);
  }

 private:
  torch::Tensor hc_head(const torch::Tensor& x) {
    auto x_float = x.to(torch::kFloat32);
    auto x_flatten = x_float.flatten(-2, -1);
    auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);
    auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
    mixes = mixes * rsqrt;
    auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
    auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
    return y.to(x.dtype());
  }

  const ModelArgs& model_args_;
  layer::RMSNorm enorm_{nullptr};
  layer::RMSNorm hnorm_{nullptr};
  layer::ReplicatedLinear e_proj_{nullptr};
  layer::ReplicatedLinear h_proj_{nullptr};
  layer::DeepseekV4DecoderLayer mtp_block_{nullptr};
  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
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

    const int32_t mtp_n_layers =
        std::max<int32_t>(model_args.num_nextn_predict_layers(), 1);
    CHECK_LE(mtp_n_layers, model_args.n_layers())
        << "deepseek_v4_mtp requires num_nextn_predict_layers <= n_layers, got "
        << mtp_n_layers << " vs " << model_args.n_layers();
    mtp_start_layer_idx_ = model_args.n_layers() - mtp_n_layers;

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
          deepseek_v4_mtp_next_power_of_two(model_args.index_head_dim());
      dsa_hadamard_ = deepseek_v4_mtp_create_hadamard_matrix(
          hadamard_dim_padded, options.dtype().toScalarType(), options.device());
    }

    build_cache_specs(context);

    mtp_layers_.reserve(mtp_n_layers);
    for (int32_t i = 0; i < mtp_n_layers; ++i) {
      const int32_t layer_index = mtp_start_layer_idx_ + i;
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

    const auto runtime_device = previous_hidden_states.device();
    tokens = deepseek_v4_mtp_maybe_to_device(tokens, runtime_device);
    positions = deepseek_v4_mtp_maybe_to_device(positions, runtime_device);

    auto mask = (positions == 0);
    if (mask.any().item<bool>()) {
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    auto attn_metadata = build_attention_metadata(modified_input_params, positions);
    prepare_for_forward(attn_metadata, runtime_device);

    CHECK_GE(static_cast<int32_t>(kv_caches.size()),
             static_cast<int32_t>(mtp_layers_.size()))
        << "deepseek_v4_mtp requires kv_caches size >= mtp layer count";

    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      const int32_t layer_id = mtp_start_layer_idx_ + static_cast<int32_t>(i);
      prepare_for_layer(attn_metadata, layer_id);
      hidden_states = mtp_layers_[i](hidden_states,
                                     previous_hidden_states,
                                     positions,
                                     attn_metadata,
                                     kv_caches[i],
                                     modified_input_params,
                                     tokens);
    }

    auto [output, _] = final_norm_(hidden_states, std::nullopt);
    return ModelOutput(output, std::nullopt);
  }

 private:
  layer::AttentionMetadata build_attention_metadata(
      const ModelInputParams& input_params,
      const torch::Tensor& positions) const {
    CHECK(!caches_info_.empty())
        << "[DeepseekV4Mtp] caches_info must not be empty";
    CHECK(!group_infos_.empty())
        << "[DeepseekV4Mtp] group_infos must not be empty";

    auto attn_metadata = layer::AttentionMetadataBuilder::build(input_params);

    auto dsa_metadata = std::make_shared<layer::DSAMetadata>();
    build_dsa_fields(input_params, positions, *dsa_metadata);

    if (attn_metadata.attn_mask.defined()) {
      dsa_metadata->attn_mask = attn_metadata.attn_mask.clone();
    }

    if (attn_metadata.mrope_cos.defined() && !dsa_metadata->cos_table.defined()) {
      dsa_metadata->cos_table = attn_metadata.mrope_cos;
    }
    if (attn_metadata.mrope_sin.defined() && !dsa_metadata->sin_table.defined()) {
      dsa_metadata->sin_table = attn_metadata.mrope_sin;
    }

    attn_metadata.dsa_metadata = std::move(dsa_metadata);

    return attn_metadata;
  }

  void build_dsa_fields(const ModelInputParams& params,
                        const torch::Tensor& positions,
                        layer::DSAMetadata& dsa) const {
    const int32_t batch_size =
        static_cast<int32_t>(params.kv_seq_lens_vec.size());

    dsa.input_positions = positions;

    build_seq_lengths(params, batch_size, dsa);

    if (dsa_cos_sin_.defined()) {
      auto cos_sin_chunks = dsa_cos_sin_.chunk(/*chunks=*/2, /*dim=*/-1);
      dsa.cos_table = cos_sin_chunks[0].contiguous();
      dsa.sin_table = cos_sin_chunks[1].contiguous();
    }

    if (positions.defined()) {
      const int64_t total_tokens = positions.numel();
      build_positions(params, batch_size, total_tokens, dsa);
    }

    if (!params.multi_block_tables.empty() && !caches_info_.empty()) {
      std::vector<torch::Tensor> active_multi_block_tables =
          params.multi_block_tables;
      int32_t manager_num =
          static_cast<int32_t>(active_multi_block_tables.size());

      CHECK_EQ(batch_size, static_cast<int32_t>(params.kv_seq_lens_vec.size()))
          << "[DeepseekV4Mtp] batch_size mismatch with kv_seq_lens_vec size.";
      CHECK_LE(manager_num, static_cast<int32_t>(group_infos_.size()))
          << "[DeepseekV4Mtp] manager_num(" << manager_num
          << ") exceeds group_infos size(" << group_infos_.size()
          << "), cannot align manager/group mapping.";
      const int32_t n_layers = static_cast<int32_t>(caches_info_.size());
      const auto& ctx_lens = params.kv_seq_lens_vec;
      int64_t total_tokens = 0;
      for (auto len : ctx_lens) total_tokens += len;

      std::vector<torch::Tensor> mgr_slots(manager_num);
      for (int32_t m = 0; m < manager_num; ++m) {
        mgr_slots[m] = expand_blocks_to_slots(active_multi_block_tables[m],
                                              group_infos_[m],
                                              ctx_lens,
                                              batch_size,
                                              total_tokens);
      }

      std::vector<torch::Tensor> proc_slots(manager_num);
      std::vector<torch::Tensor> proc_bt(manager_num);
      for (int32_t m = 0; m < manager_num; ++m) {
        process_group(active_multi_block_tables[m],
                      mgr_slots[m],
                      group_infos_[m],
                      ctx_lens,
                      params.q_seq_lens_vec,
                      batch_size,
                      total_tokens,
                      proc_bt[m],
                      proc_slots[m]);
      }

      const torch::Device target_device =
          positions.defined() ? positions.device() : torch::Device(torch::kCPU);
      if (!target_device.is_cpu()) {
        for (int32_t m = 0; m < manager_num; ++m) {
          if (proc_bt[m].defined() && proc_bt[m].device() != target_device) {
            proc_bt[m] = proc_bt[m].to(target_device);
          }
          if (proc_slots[m].defined() &&
              proc_slots[m].device() != target_device) {
            proc_slots[m] = proc_slots[m].to(target_device);
          }
        }
      }

      dsa.block_tables.resize(n_layers);
      dsa.slot_mappings.resize(n_layers);
      for (int32_t lid = 0; lid < n_layers; ++lid) {
        const auto& lci = caches_info_[lid];
        dsa.block_tables[lid].resize(lci.size());
        dsa.slot_mappings[lid].resize(lci.size());
        for (size_t ci = 0; ci < lci.size(); ++ci) {
          int32_t gid = lci[ci].group_id;
          if (gid < manager_num) {
            dsa.block_tables[lid][ci] = proc_bt[gid];
            dsa.slot_mappings[lid][ci] = proc_slots[gid];
          }
        }
      }
    }

    dsa.caches_info = &caches_info_;
  }

  static torch::Tensor expand_blocks_to_slots(
      const torch::Tensor& block_table,
      const DSAGroupInfo& gi,
      const std::vector<int>& ctx_lens,
      int32_t batch_size,
      int64_t total_tokens) {
    const int32_t bs = gi.block_size;
    auto slots = torch::full({total_tokens}, -1, torch::kInt32);
    auto slots_acc = slots.accessor<int32_t, 1>();
    auto bt_acc = block_table.accessor<int32_t, 2>();
    const int32_t max_blocks = block_table.size(1);

    int64_t start_idx = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      int64_t token_len = ctx_lens[seq];
      int64_t slot_num = compute_slot_num(gi, token_len);

      int64_t filled = 0;
      for (int32_t blk = 0; blk < max_blocks && filled < slot_num; ++blk) {
        int32_t block_id = bt_acc[seq][blk];
        if (block_id < 0) break;
        for (int32_t off = 0; off < bs && filled < slot_num; ++off) {
          slots_acc[start_idx + filled] =
              static_cast<int32_t>(static_cast<int64_t>(block_id) * bs + off);
          ++filled;
        }
      }
      start_idx += token_len;
    }
    return slots;
  }

  static int64_t compute_slot_num(const DSAGroupInfo& gi, int64_t token_len) {
    if (gi.type == DSACacheType::TOKEN) {
      return token_len / gi.ratio;
    }
    const int32_t bs = gi.block_size;
    if (token_len > bs) {
      return token_len % bs + bs;
    }
    int64_t n = token_len % bs;
    return (n == 0 && token_len > 0) ? bs : n;
  }

  static void process_group(const torch::Tensor& raw_bt,
                           const torch::Tensor& raw_slots,
                           const DSAGroupInfo& gi,
                           const std::vector<int>& ctx_lens,
                           const std::vector<int>& q_lens_vec,
                           int32_t batch_size,
                           int64_t total_tokens,
                           torch::Tensor& out_bt,
                           torch::Tensor& out_slots) {
    std::vector<int> q_lens;
    if (static_cast<int32_t>(q_lens_vec.size()) == batch_size) {
      q_lens = q_lens_vec;
    } else {
      q_lens.assign(batch_size, 1);
    }

    if (gi.type == DSACacheType::TOKEN) {
      process_token_group(raw_bt, raw_slots, gi.ratio, ctx_lens, q_lens,
                          batch_size, total_tokens, out_bt, out_slots);
    } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
      process_swa_group(raw_bt, raw_slots, gi.block_size, ctx_lens, q_lens,
                        batch_size, out_bt, out_slots);
    } else {
      out_slots =
          torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);
      out_bt = raw_bt;
    }
  }

  static void process_token_group(const torch::Tensor& raw_bt,
                                 const torch::Tensor& raw_slots,
                                 int32_t ratio,
                                 const std::vector<int>& ctx_lens,
                                 const std::vector<int>& q_lens,
                                 int32_t batch_size,
                                 int64_t total_tokens,
                                 torch::Tensor& out_bt,
                                 torch::Tensor& out_slots) {
    int64_t committed_rows = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
      const int64_t q_len =
          std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
      const int64_t prev_ctx_len = ctx_len - q_len;
      committed_rows += ctx_len / ratio - prev_ctx_len / ratio;
    }

    auto out_slots_tensor = torch::empty({committed_rows}, raw_slots.options());
    auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
    auto raw_slots_acc = raw_slots.accessor<int32_t, 1>();

    int64_t start_idx = 0;
    int64_t write_idx = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
      const int64_t q_len =
          std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
      const int64_t prev_ctx_len = ctx_len - q_len;
      const int64_t prev_committed = prev_ctx_len / ratio;
      const int64_t committed = ctx_len / ratio;
      const int64_t new_committed = committed - prev_committed;
      for (int64_t i = 0; i < new_committed; ++i) {
        out_slots_acc[write_idx++] =
            raw_slots_acc[start_idx + prev_committed + i];
      }
      start_idx += ctx_len;
    }

    out_slots = out_slots_tensor;
    out_bt = raw_bt;
  }

  static void process_swa_group(const torch::Tensor& raw_bt,
                               const torch::Tensor& raw_slots,
                               int32_t block_size,
                               const std::vector<int>& ctx_lens,
                               const std::vector<int>& q_lens,
                               int32_t batch_size,
                               torch::Tensor& out_bt,
                               torch::Tensor& out_slots) {
    int64_t query_total_tokens = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      query_total_tokens += std::clamp<int64_t>(
          static_cast<int64_t>(q_lens[seq]), 0, ctx_lens[seq]);
    }

    auto out_slots_tensor =
        torch::full({query_total_tokens}, -1, raw_slots.options());
    auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
    auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
    const int64_t max_blocks = raw_bt.size(1);
    const int64_t block_size_i64 = static_cast<int64_t>(block_size);

    auto slot_for_position = [&](int32_t seq, int64_t pos) -> int32_t {
      if (max_blocks <= 0) {
        return -1;
      }
      const int64_t block_idx = (pos / block_size_i64) % max_blocks;
      const int32_t block_id = raw_bt_acc[seq][block_idx];
      if (block_id < 0) {
        return -1;
      }
      const int64_t block_offset = pos % block_size_i64;
      return static_cast<int32_t>(
          static_cast<int64_t>(block_id) * block_size_i64 + block_offset);
    };

    int64_t write_idx = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
      const int64_t q_len =
          std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
      const int64_t q_start = ctx_len - q_len;
      for (int64_t i = 0; i < q_len; ++i) {
        out_slots_acc[write_idx++] = slot_for_position(seq, q_start + i);
      }
    }

    out_slots = out_slots_tensor;

    int32_t current_cols = raw_bt.size(1);
    int32_t max_dst_len = 0;
    std::vector<int32_t> dst_lens(batch_size);
    for (int32_t s = 0; s < batch_size; ++s) {
      dst_lens[s] = static_cast<int32_t>(
          std::ceil(static_cast<double>(ctx_lens[s]) / block_size));
      max_dst_len = std::max(max_dst_len, dst_lens[s]);
    }
    max_dst_len = std::max(max_dst_len, current_cols);

    auto new_bt = torch::zeros({batch_size, max_dst_len}, raw_bt.options());
    auto new_acc = new_bt.accessor<int32_t, 2>();
    auto old_acc = raw_bt.accessor<int32_t, 2>();

    for (int32_t s = 0; s < batch_size; ++s) {
      const int32_t retained_cols = std::min(current_cols, dst_lens[s]);
      const int32_t start_col = dst_lens[s] - retained_cols;
      for (int32_t j = 0; j < retained_cols; ++j) {
        const int32_t logical_col = start_col + j;
        const int32_t physical_col = logical_col % current_cols;
        new_acc[s][logical_col] = old_acc[s][physical_col];
      }
    }
    out_bt = new_bt;
  }

  static void build_seq_lengths(const ModelInputParams& params,
                                int32_t batch_size,
                                layer::DSAMetadata& dsa_metadata) {
    auto kv_lens =
        torch::tensor(std::vector<int32_t>(params.kv_seq_lens_vec.begin(),
                                           params.kv_seq_lens_vec.end()),
                      torch::kInt32);
    dsa_metadata.seq_lens = kv_lens;
    dsa_metadata.actual_seq_lengths_kv = kv_lens;

    torch::Tensor q_lens;
    if (static_cast<int32_t>(params.q_seq_lens_vec.size()) == batch_size) {
      q_lens = torch::tensor(std::vector<int32_t>(params.q_seq_lens_vec.begin(),
                                                  params.q_seq_lens_vec.end()),
                             torch::kInt32);
    } else if (params.batch_forward_type.no_decode()) {
      q_lens = kv_lens;
    } else {
      q_lens = torch::ones({batch_size}, torch::kInt32);
    }

    auto cumsum = torch::cumsum(q_lens, /*dim=*/0, /*dtype=*/torch::kInt32);
    dsa_metadata.actual_seq_lengths_query =
        torch::cat({torch::zeros({1}, torch::kInt32), cumsum});
    dsa_metadata.seq_lens_q = q_lens;

    auto int_options = torch::TensorOptions().dtype(torch::kInt32);
    if (kv_lens.numel() > 0) {
      dsa_metadata.max_seqlen_kv = torch::max(kv_lens).to(torch::kInt32);
    } else {
      dsa_metadata.max_seqlen_kv = torch::zeros({1}, int_options);
    }

    if (q_lens.numel() > 0) {
      dsa_metadata.max_seqlen_q = torch::max(q_lens).to(torch::kInt32);
    } else {
      dsa_metadata.max_seqlen_q = torch::zeros({1}, int_options);
    }
  }

  static void build_positions(const ModelInputParams& params,
                              int32_t batch_size,
                              int64_t total_tokens,
                              layer::DSAMetadata& dsa_metadata) {
    (void)params;
    (void)total_tokens;
    if (!dsa_metadata.input_positions.defined()) return;

    auto input_positions = dsa_metadata.input_positions;
    int64_t num_tokens = input_positions.size(0);

    auto c4_mask = ((input_positions + 1) % 4).eq(0);
    auto c4_pos = input_positions.index({c4_mask});
    c4_pos = (c4_pos + 1) - 4;
    int64_t c4_target = std::min(num_tokens, num_tokens / 4 + batch_size);
    int64_t c4_pad_right = c4_target - c4_pos.size(0);
    if (c4_pad_right > 0) {
      dsa_metadata.c4_pad_positions =
          torch::cat({c4_pos, torch::zeros({c4_pad_right}, c4_pos.options())});
    } else {
      dsa_metadata.c4_pad_positions = c4_pos.slice(0, 0, c4_target);
    }

    auto c128_mask = ((input_positions + 1) % 128).eq(0);
    auto c128_pos = input_positions.index({c128_mask});
    c128_pos = (c128_pos + 1) - 128;
    int64_t c128_target = std::min(num_tokens, num_tokens / 128 + batch_size);
    int64_t c128_pad_right = c128_target - c128_pos.size(0);
    if (c128_pad_right > 0) {
      dsa_metadata.c128_pad_positions = torch::cat(
          {c128_pos, torch::zeros({c128_pad_right}, c128_pos.options())});
    } else {
      dsa_metadata.c128_pad_positions = c128_pos.slice(0, 0, c128_target);
    }
  }

  void build_cache_specs(const ModelContext& context) {
    const auto& model_args = context.get_model_args();
    const auto& compress_ratios = model_args.compress_ratios();
    const int32_t window_size = model_args.window_size();
    const int32_t base_block_size = 128;

    struct DSAGroupKey {
      int32_t ratio;
      DSACacheType type;
      int32_t block_size;
      bool operator==(const DSAGroupKey& o) const {
        return ratio == o.ratio && type == o.type && block_size == o.block_size;
      }
    };

    struct DSAGroupKeyHash {
      size_t operator()(const DSAGroupKey& k) const {
        size_t h = std::hash<int32_t>()(k.ratio);
        h ^= std::hash<int32_t>()(static_cast<int32_t>(k.type)) << 16;
        h ^= std::hash<int32_t>()(k.block_size) << 8;
        return h;
      }
    };

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    auto register_group = [&](DSACacheType type, int32_t ratio,
                              int32_t block_size) -> int32_t {
      DSAGroupKey key{ratio, type, block_size};
      auto it = group_key_map.find(key);
      if (it != group_key_map.end()) {
        return it->second;
      }
      const int32_t gid = static_cast<int32_t>(group_infos_.size());
      group_key_map.emplace(key, gid);
      group_infos_.push_back({type, ratio, block_size});
      return gid;
    };

    register_group(DSACacheType::SLIDING_WINDOW, 1, window_size);
    for (const auto ratio : compress_ratios) {
      const int32_t cr = deepseek_v4_mtp_normalize_compress_ratio(ratio);
      if (cr == 4 || cr == 128) {
        register_group(DSACacheType::TOKEN, cr, base_block_size);
      }
    }

    caches_info_.resize(model_args.n_layers());

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                        ? compress_ratios[layer_id]
                        : 1;
      cr = deepseek_v4_mtp_normalize_compress_ratio(cr);

      struct CacheEntry {
        DSACacheType type;
        int32_t ratio;
        int32_t block_size;
      };
      std::vector<CacheEntry> layer_caches;

      if (cr == 1) {
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
      } else if (cr == 4) {
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      } else if (cr == 128) {
        layer_caches.push_back(
            {DSACacheType::TOKEN, 128, base_block_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back(
            {DSACacheType::SLIDING_WINDOW, 1, window_size});
      }

      for (const auto& ce : layer_caches) {
        const int32_t gid = register_group(ce.type, ce.ratio, ce.block_size);
        caches_info_[layer_id].push_back({gid, ce.type, ce.ratio, ce.block_size});
      }
    }
  }

  void prepare_for_forward(layer::AttentionMetadata& attn_metadata,
                           const torch::Device& runtime_device) const {
    CHECK(attn_metadata.dsa_metadata)
        << "[DeepseekV4Mtp] attn_metadata.dsa_metadata must be populated";

    auto& dsa = *(attn_metadata.dsa_metadata);

    dsa.seq_lens = deepseek_v4_mtp_maybe_to_device(dsa.seq_lens, runtime_device);
    dsa.seq_lens_q =
        deepseek_v4_mtp_maybe_to_device(dsa.seq_lens_q, runtime_device);
    dsa.actual_seq_lengths_query =
        deepseek_v4_mtp_maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
    dsa.actual_seq_lengths_kv =
        deepseek_v4_mtp_maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
    dsa.max_seqlen_q =
        deepseek_v4_mtp_maybe_to_device(dsa.max_seqlen_q, runtime_device);
    dsa.max_seqlen_kv =
        deepseek_v4_mtp_maybe_to_device(dsa.max_seqlen_kv, runtime_device);
    dsa.input_positions =
        deepseek_v4_mtp_maybe_to_device(dsa.input_positions, runtime_device);
    dsa.c4_pad_positions =
        deepseek_v4_mtp_maybe_to_device(dsa.c4_pad_positions, runtime_device);
    dsa.c128_pad_positions =
        deepseek_v4_mtp_maybe_to_device(dsa.c128_pad_positions, runtime_device);

    for (auto& layer_block_tables : dsa.block_tables) {
      for (auto& block_table : layer_block_tables) {
        block_table = deepseek_v4_mtp_maybe_to_device(block_table, runtime_device);
      }
    }
    for (auto& layer_slot_mappings : dsa.slot_mappings) {
      for (auto& slot_mapping : layer_slot_mappings) {
        slot_mapping = deepseek_v4_mtp_maybe_to_device(slot_mapping, runtime_device);
      }
    }

    if (dsa_hadamard_.defined()) {
      dsa.hadamard = deepseek_v4_mtp_maybe_to_device(dsa_hadamard_, runtime_device);
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

    dsa.cos = deepseek_v4_mtp_maybe_to_device(dsa.cos, runtime_device);
    dsa.sin = deepseek_v4_mtp_maybe_to_device(dsa.sin, runtime_device);
    dsa.c4_cos = deepseek_v4_mtp_maybe_to_device(dsa.c4_cos, runtime_device);
    dsa.c4_sin = deepseek_v4_mtp_maybe_to_device(dsa.c4_sin, runtime_device);
    dsa.c128_cos =
        deepseek_v4_mtp_maybe_to_device(dsa.c128_cos, runtime_device);
    dsa.c128_sin =
        deepseek_v4_mtp_maybe_to_device(dsa.c128_sin, runtime_device);
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
        deepseek_v4_mtp_maybe_to_device(dsa.actual_seq_lengths_query, metadata_device);
    dsa.actual_seq_lengths_kv =
        deepseek_v4_mtp_maybe_to_device(dsa.actual_seq_lengths_kv, metadata_device);
    dsa.seq_lens_q =
        deepseek_v4_mtp_maybe_to_device(dsa.seq_lens_q, metadata_device);
    dsa.seq_lens =
        deepseek_v4_mtp_maybe_to_device(dsa.seq_lens, metadata_device);
    dsa.max_seqlen_q =
        deepseek_v4_mtp_maybe_to_device(dsa.max_seqlen_q, metadata_device);
    dsa.max_seqlen_kv =
        deepseek_v4_mtp_maybe_to_device(dsa.max_seqlen_kv, metadata_device);

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
        deepseek_v4_mtp_normalize_compress_ratio(
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
  int32_t mtp_start_layer_idx_ = 0;
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