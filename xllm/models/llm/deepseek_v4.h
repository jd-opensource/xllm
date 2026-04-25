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

#include "core/framework/state_dict/utils.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/dsa_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "layers/npu/deepseek_v4_rotary_embedding.h"
#include "llm_model_base.h"

namespace xllm {

inline int64_t deepseek_v4_next_power_of_two(int64_t n) {
  int64_t value = 1;
  while (value < n) {
    value <<= 1;
  }
  return value;
}

inline torch::Tensor deepseek_v4_create_hadamard_matrix(
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

inline std::string deepseek_v4_format_bytes(int64_t bytes) {
  static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit_idx = 0;
  while (value >= 1024.0 && unit_idx < 4) {
    value /= 1024.0;
    ++unit_idx;
  }

  std::ostringstream os;
  if (value < 10.0 && unit_idx > 0) {
    os << std::fixed << std::setprecision(2);
  } else if (value < 100.0 && unit_idx > 0) {
    os << std::fixed << std::setprecision(1);
  } else {
    os << std::fixed << std::setprecision(0);
  }
  os << value << " " << kUnits[unit_idx];
  return os.str();
}

inline torch::Tensor maybe_to_device(const torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

// Group key: (ratio, type, block_size) -> group_id
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

class DeepseekV4ModelImpl
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = model_args.hc_mult();
    hc_eps_ = static_cast<double>(model_args.hc_eps());
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    num_heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    head_dim_ = model_args.o_lora_rank() + model_args.qk_rope_head_dim();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DSV4][Init] n_heads must be divisible by local tp size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_
        << ", world_size=" << parallel_args.world_size()
        << ", dp_size=" << parallel_args.dp_size();
    tp_num_heads_ = num_heads_ / dp_local_tp_size_;
    window_size_ = model_args.window_size();
    index_n_heads_ = model_args.index_n_heads();
    index_head_dim_ = model_args.index_head_dim();
    index_topk_ = model_args.index_topk();

    const int64_t hc_dim = hc_mult_ * model_args.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ =
        register_parameter("hc_head_fn",
                           torch::empty({hc_mult_, hc_dim}, hc_options),
                           /*requires_grad=*/false);
    hc_head_base_ = register_parameter("hc_head_base",
                                       torch::empty({hc_mult_}, hc_options),
                                       /*requires_grad=*/false);
    hc_head_scale_ = register_parameter("hc_head_scale",
                                        torch::empty({1}, hc_options),
                                        /*requires_grad=*/false);

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
              /*extrapolation_factor=*/model_args.rope_extrapolation_factor(),
              /*beta_fast=*/model_args.beta_fast(),
              /*beta_slow=*/model_args.beta_slow(),
              /*attn_factor=*/model_args.rope_scaling_attn_factor(),
              /*mscale=*/model_args.rope_scaling_mscale(),
              /*mscale_all_dim=*/model_args.rope_scaling_mscale_all_dim(),
              /*original_max_position_embeddings=*/original_max_pos,
              options);
      dsa_cos_sin_ = dsa_rotary_embedding_->get_cos_sin_cache("default");
    }

    if (model_args.index_head_dim() > 0) {
      auto hadamard_dim_padded =
          deepseek_v4_next_power_of_two(model_args.index_head_dim());
      dsa_hadamard_ =
          deepseek_v4_create_hadamard_matrix(hadamard_dim_padded,
                                             options.dtype().toScalarType(),
                                             options.device());
    }

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::DeepseekV4DecoderLayer(context, i);
      layers_.push_back(layer);
    }

    // Build DSA caches_info from compress_ratios
    const auto& compress_ratios = model_args.compress_ratios();
    const int32_t window_size = model_args.window_size();
    const int32_t base_block_size = 128;  // default block size

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    auto register_group =
        [&](DSACacheType type, int32_t ratio, int32_t block_size) -> int32_t {
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

    // Keep DSA group ids consistent with BlockManagerPool manager order:
    // 1) sliding-window manager first
    // 2) token managers in first-seen compress_ratio order
    register_group(DSACacheType::SLIDING_WINDOW, 1, window_size);
    for (const auto ratio : compress_ratios) {
      if (ratio == 4 || ratio == 128) {
        register_group(DSACacheType::TOKEN, ratio, base_block_size);
      }
    }

    caches_info_.resize(model_args.n_layers());

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                       ? compress_ratios[layer_id]
                       : 1;
      // Build per-layer cache specs based on compress_ratio
      struct CacheEntry {
        DSACacheType type;
        int32_t ratio;
        int32_t block_size;
      };
      std::vector<CacheEntry> layer_caches;

      if (cr == 1) {
        // C1: 1 cache (swa)
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      } else if (cr == 4) {
        // C4: 8 caches
        // compress_kv(TOKEN,4,128), compress_index(TOKEN,4,128),
        // swa(SW,1,window), kv_state(SW,1,window), score_state(SW,1,window),
        // idx_kv_state(SW,1,window), idx_score_state(SW,1,window),
        // indexer_scale(TOKEN,4,128)
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      } else if (cr == 128) {
        // C128: 4 caches
        // compress_kv(TOKEN,128,128), swa(SW,1,window),
        // kv_state(SW,1,window), score_state(SW,1,window)
        layer_caches.push_back({DSACacheType::TOKEN, 128, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      }

      for (const auto& ce : layer_caches) {
        const int32_t gid = register_group(ce.type, ce.ratio, ce.block_size);
        caches_info_[layer_id].push_back(
            {gid, ce.type, ce.ratio, ce.block_size});
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    LlmModelImplBase<layer::DeepseekV4DecoderLayer>::load_state_dict(
        state_dict);
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix("embed."));
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  void log_weight_mem_stats() const { log_layer_weight_mem_stats(); }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }

    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h =
        inputs_embeds.defined() ? inputs_embeds : embed_tokens_(tokens);

    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    // Keep runtime inputs on the same accelerator device.
    const auto runtime_device = h.device();
    tokens = maybe_to_device(tokens, runtime_device);
    positions = maybe_to_device(positions, runtime_device);

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    // DP helper: keep zero entries at least 1 to avoid empty slices/padding
    // in xllm DP utilities. DeepSeek V4 not use DP today.
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::DSAMetadataBuilder::build(modified_input_params,
                                               positions,
                                               dsa_cos_sin_,
                                               caches_info_,
                                               group_infos_));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);

    if (attn_metadata.dsa_metadata) {
      auto& dsa = *(attn_metadata.dsa_metadata);

      // DSAMetadataBuilder may create several length/position tensors on CPU;
      // move all operator inputs to runtime device before invoking NPU kernels.
      dsa.seq_lens = maybe_to_device(dsa.seq_lens, runtime_device);
      dsa.seq_lens_q = maybe_to_device(dsa.seq_lens_q, runtime_device);
      dsa.actual_seq_lengths_query =
          maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
      dsa.actual_seq_lengths_kv =
          maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
      dsa.max_seqlen_q = maybe_to_device(dsa.max_seqlen_q, runtime_device);
      dsa.max_seqlen_kv = maybe_to_device(dsa.max_seqlen_kv, runtime_device);
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

      if (dsa_rotary_embedding_) {
        std::unordered_map<std::string, torch::Tensor> positions_map;
        // Avoid stale group tensors when attn_metadata is reused across runs.
        dsa.cos = torch::Tensor();
        dsa.sin = torch::Tensor();
        dsa.c4_cos = torch::Tensor();
        dsa.c4_sin = torch::Tensor();
        dsa.c128_cos = torch::Tensor();
        dsa.c128_sin = torch::Tensor();

        auto append_group_positions = [&positions_map](
                                          const std::string& group,
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
      }

      if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
        dsa.start_pos =
            (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
      }

      build_precomputed_metadata(dsa);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (attn_metadata.dsa_metadata) {
        auto& dsa = *(attn_metadata.dsa_metadata);
        const int32_t layer_id = static_cast<int32_t>(i);
        dsa.layer_id = layer_id;

        if (layer_id < static_cast<int32_t>(dsa.block_tables.size()) &&
            layer_id < static_cast<int32_t>(dsa.slot_mappings.size()) &&
            !dsa.block_tables[layer_id].empty() &&
            !dsa.slot_mappings[layer_id].empty()) {
          size_t attn_cache_idx = 0;
          if (layer_id < static_cast<int32_t>(caches_info_.size())) {
            const auto& layer_caches = caches_info_[layer_id];
            for (size_t cache_idx = 0; cache_idx < layer_caches.size();
                 ++cache_idx) {
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

      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params,
                     tokens);
    }
    h = hc_head(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  static c10::optional<torch::Tensor> as_optional_tensor(
      const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
      return c10::optional<torch::Tensor>(tensor);
    }
    return c10::nullopt;
  }

  static int64_t tensor_max_or_zero(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
      return 0;
    }
    return tensor.max().item<int64_t>();
  }

  static int64_t pick_max_seqlen(const torch::Tensor& max_seqlen_tensor,
                                 const torch::Tensor& fallback_tensor) {
    if (max_seqlen_tensor.defined() && max_seqlen_tensor.numel() > 0) {
      return max_seqlen_tensor.max().item<int64_t>();
    }
    return tensor_max_or_zero(fallback_tensor);
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
    dsa.seq_lens_q = maybe_to_device(dsa.seq_lens_q, metadata_device);
    dsa.seq_lens = maybe_to_device(dsa.seq_lens, metadata_device);
    dsa.max_seqlen_q = maybe_to_device(dsa.max_seqlen_q, metadata_device);
    dsa.max_seqlen_kv = maybe_to_device(dsa.max_seqlen_kv, metadata_device);

    if (!dsa.actual_seq_lengths_query.defined() ||
        !dsa.actual_seq_lengths_kv.defined()) {
      return;
    }

    const int64_t batch_size =
        std::max<int64_t>(dsa.actual_seq_lengths_kv.size(0), 1);
    const int64_t max_seqlen_q =
        pick_max_seqlen(dsa.max_seqlen_q, dsa.seq_lens_q);
    const int64_t max_seqlen_kv =
        pick_max_seqlen(dsa.max_seqlen_kv, dsa.actual_seq_lengths_kv);
    const int64_t ori_win_left = std::max<int64_t>(window_size_ - 1, 0);
    const int64_t sparse_topk = std::max<int64_t>(index_topk_, 1);

    xllm::kernel::SparseAttnSharedkvMetadataParams c1_params;
    c1_params.num_heads_q = tp_num_heads_;
    c1_params.num_heads_kv = 1;
    c1_params.head_dim = head_dim_;
    c1_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c1_params.cu_seqlens_ori_kv = c10::nullopt;
    c1_params.cu_seqlens_cmp_kv = c10::nullopt;
    c1_params.seqused_q = as_optional_tensor(dsa.seq_lens_q);
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
    c1_params.layout_kv = "PA_ND";
    c1_params.has_ori_kv = true;
    c1_params.has_cmp_kv = false;
    dsa.c1_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c1_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c4_params;
    c4_params.num_heads_q = tp_num_heads_;
    c4_params.num_heads_kv = 1;
    c4_params.head_dim = head_dim_;
    c4_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c4_params.cu_seqlens_ori_kv = c10::nullopt;
    c4_params.cu_seqlens_cmp_kv = c10::nullopt;
    c4_params.seqused_q = as_optional_tensor(dsa.seq_lens_q);
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
    c4_params.layout_kv = "PA_ND";
    c4_params.has_ori_kv = true;
    c4_params.has_cmp_kv = true;
    dsa.c4_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c4_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c128_params;
    c128_params.num_heads_q = tp_num_heads_;
    c128_params.num_heads_kv = 1;
    c128_params.head_dim = head_dim_;
    c128_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c128_params.cu_seqlens_ori_kv = c10::nullopt;
    c128_params.cu_seqlens_cmp_kv = c10::nullopt;
    c128_params.seqused_q = as_optional_tensor(dsa.seq_lens_q);
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
    c128_params.layout_kv = "PA_ND";
    c128_params.has_ori_kv = true;
    c128_params.has_cmp_kv = true;
    dsa.c128_metadata =
        xllm::kernel::sparse_attn_sharedkv_metadata(c128_params);

    torch::Tensor query_lens;
    if (dsa.actual_seq_lengths_query.defined() &&
        dsa.actual_seq_lengths_query.dim() > 0 &&
        dsa.actual_seq_lengths_query.size(0) > 1) {
      query_lens = dsa.actual_seq_lengths_query.slice(
          /*dim=*/0,
          /*start=*/1,
          /*end=*/dsa.actual_seq_lengths_query.size(0));
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
        << "[DSV4][PrecomputeMetadata] index/global heads must be divisible by "
           "local tp size. global_index_num_heads="
        << global_index_num_heads << ", local_tp_size=" << dp_local_tp_size_;
    const int64_t index_num_heads =
        std::max<int64_t>(global_index_num_heads / dp_local_tp_size_, 1);
    const int64_t index_head_dim =
        std::max<int64_t>(index_head_dim_ > 0 ? index_head_dim_ : head_dim_, 1);
    const int64_t qli_batch_size = std::max<int64_t>(key_lens.size(0), 1);
    const int64_t qli_max_seqlen_q =
        pick_max_seqlen(dsa.max_seqlen_q, query_lens);
    const int64_t qli_max_seqlen_k =
        pick_max_seqlen(dsa.max_seqlen_kv, key_lens);

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

  void log_layer_weight_mem_stats() const {
    int64_t total_bytes = 0;
    int64_t attn_bytes = 0;
    int64_t expert_bytes = 0;
    int64_t hc_bytes = 0;
    int64_t other_bytes = 0;

    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto stats = layers_[i]->get_weight_mem_stats();
      total_bytes += stats.total_bytes;
      attn_bytes += stats.attn_bytes;
      expert_bytes += stats.expert_bytes;
      hc_bytes += stats.hc_bytes;
      other_bytes += stats.other_bytes;

      LOG(INFO) << "[WEIGHT_MEM][DeepseekV4] layer=" << i
                << " total=" << deepseek_v4_format_bytes(stats.total_bytes)
                << " (" << stats.total_bytes << " B)"
                << ", attn=" << deepseek_v4_format_bytes(stats.attn_bytes)
                << " (" << stats.attn_bytes << " B)"
                << ", expert=" << deepseek_v4_format_bytes(stats.expert_bytes)
                << " (" << stats.expert_bytes << " B)"
                << ", hc_=" << deepseek_v4_format_bytes(stats.hc_bytes) << " ("
                << stats.hc_bytes << " B)"
                << ", other=" << deepseek_v4_format_bytes(stats.other_bytes)
                << " (" << stats.other_bytes << " B)";
    }

    LOG(INFO) << "[WEIGHT_MEM][DeepseekV4][Summary] layers=" << layers_.size()
              << ", total=" << deepseek_v4_format_bytes(total_bytes) << " ("
              << total_bytes << " B)"
              << ", attn=" << deepseek_v4_format_bytes(attn_bytes) << " ("
              << attn_bytes << " B)"
              << ", expert=" << deepseek_v4_format_bytes(expert_bytes) << " ("
              << expert_bytes << " B)"
              << ", hc_=" << deepseek_v4_format_bytes(hc_bytes) << " ("
              << hc_bytes << " B)"
              << ", other=" << deepseek_v4_format_bytes(other_bytes) << " ("
              << other_bytes << " B)";
  }

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

  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_hadamard_;
  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;

  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  int64_t num_heads_ = 0;
  int64_t tp_num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
  int64_t head_dim_ = 0;
  int64_t window_size_ = 128;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 512;

  // DSA cache group info: built once at model init from compress_ratios
  // caches_info_[layer_id] = vector of DSACacheInfo for each cache in that
  // layer
  std::vector<std::vector<DSACacheInfo>> caches_info_;
  // group_infos_[group_id] = DSAGroupInfo
  std::vector<DSAGroupInfo> group_infos_;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<DeepseekV4Model>::load_model(std::move(loader),
                                                        std::move(prefix));
    this->model_->log_weight_mem_stats();
  }
};
TORCH_MODULE(DeepseekV4ForCausalLM);

inline void load_deepseek_v4_model_args(const JsonReader& json,
                                        ModelArgs* args) {
  // --------------------------------------------------------------------------
  // Base args (shared/common HF CausalLM args).
  // NOTE: These are intentionally kept here even if they are not DSv4-specific.
  // --------------------------------------------------------------------------
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("norm_eps", 1e-6f);
  });

  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR(rope_head_dim, "qk_rope_head_dim", 64);

  // LoRA / groups
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(o_groups, "o_groups", 8);

  // --------------------------------------------------------------------------
  // DeepSeek V4 args.
  // --------------------------------------------------------------------------
  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG_OR(window_size, "window_size", 128);

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_activated_experts, "n_activated_experts", 6);
  LOAD_ARG_OR_FUNC(num_experts_per_tok, "num_experts_per_tok", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(route_scale, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(score_func, "scoring_func", "sqrtsoftplus");

  // Indexer
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  // HC / DSA helpers
  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
}

struct DeepseekV4ArgsPolicy {
  std::unordered_set<int32_t> supported_compress_ratios;
  std::unordered_set<std::string> supported_score_funcs;
  int32_t default_compress_ratio = 1;
};

enum class DeepseekV4PolicyPreset {
  kDefault,
};

inline DeepseekV4ArgsPolicy build_deepseek_v4_args_policy(
    DeepseekV4PolicyPreset preset) {
  DeepseekV4ArgsPolicy policy;
  switch (preset) {
    case DeepseekV4PolicyPreset::kDefault:
      policy.supported_compress_ratios = {1, 4, 128};
      policy.supported_score_funcs = {"softmax", "sigmoid", "sqrtsoftplus"};
      policy.default_compress_ratio = 1;
      return policy;
  }
  return policy;
}

inline void process_deepseek_v4_args(ModelArgs* args,
                                     const DeepseekV4ArgsPolicy& policy) {
  // Keep alias fields consistent after loading.
  SET_ARG(n_activated_experts, args->num_experts_per_tok());

  // Align with MindIE behavior: missing tail ratios default to non-compressed.
  if (args->n_layers() > 0 &&
      static_cast<int64_t>(args->compress_ratios().size()) < args->n_layers()) {
    args->compress_ratios().resize(static_cast<size_t>(args->n_layers()),
                                   policy.default_compress_ratio);
  }

  // Build stop token set from eos for runtime usage.
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
}

inline void validate_deepseek_v4_args(const ModelArgs& args,
                                      const DeepseekV4ArgsPolicy& policy) {
  CHECK(!policy.supported_compress_ratios.empty())
      << "deepseek_v4 internal supported_compress_ratios must not be empty";
  CHECK(policy.supported_compress_ratios.count(policy.default_compress_ratio) >
        0)
      << "deepseek_v4 config default_compress_ratio("
      << policy.default_compress_ratio
      << ") must exist in supported_compress_ratios";
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v4 config n_layers/num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v4 config compress_ratios size must be >= n_layers after "
         "normalization, got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (int64_t i = 0; i < args.n_layers(); ++i) {
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(i)];
    CHECK(policy.supported_compress_ratios.count(ratio) > 0)
        << "deepseek_v4 config compress_ratios[" << i
        << "] must be in supported_compress_ratios, got " << ratio;
  }
  CHECK_GT(args.window_size(), 0)
      << "deepseek_v4 config window_size must be > 0, got "
      << args.window_size();
  CHECK_GT(args.n_routed_experts(), 0)
      << "deepseek_v4 config n_routed_experts must be > 0, got "
      << args.n_routed_experts();
  CHECK_GT(args.n_activated_experts(), 0)
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "> 0, got "
      << args.n_activated_experts();
  CHECK_LE(args.n_activated_experts(), args.n_routed_experts())
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "<= n_routed_experts, got "
      << args.n_activated_experts() << " vs " << args.n_routed_experts();
  CHECK_GE(args.n_hash_layers(), 0)
      << "deepseek_v4 config num_hash_layers/n_hash_layers must be >= 0, got "
      << args.n_hash_layers();
  CHECK_GT(args.route_scale(), 0.0f)
      << "deepseek_v4 config routed_scaling_factor/route_scale must be > 0, "
         "got "
      << args.route_scale();
  CHECK(!args.score_func().empty())
      << "deepseek_v4 config scoring_func/score_func must not be empty";
  {
    std::string score_func = args.score_func();
    std::transform(
        score_func.begin(),
        score_func.end(),
        score_func.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    CHECK(policy.supported_score_funcs.count(score_func) > 0)
        << "deepseek_v4 config scoring_func/score_func must be in "
        << absl::StrJoin(policy.supported_score_funcs, ", ") << ", got "
        << args.score_func();
  }
  CHECK_GT(args.index_head_dim(), 0)
      << "deepseek_v4 config index_head_dim must be > 0, got "
      << args.index_head_dim();
  CHECK_GT(args.index_n_heads(), 0)
      << "deepseek_v4 config index_n_heads must be > 0, got "
      << args.index_n_heads();
  CHECK_GT(args.index_topk(), 0)
      << "deepseek_v4 config index_topk must be > 0, got " << args.index_topk();
  CHECK_GT(args.hc_mult(), 0)
      << "deepseek_v4 config hc_mult must be > 0, got " << args.hc_mult();
  CHECK_GE(args.hc_sinkhorn_iters(), 0)
      << "deepseek_v4 config hc_sinkhorn_iters must be >= 0, got "
      << args.hc_sinkhorn_iters();
  CHECK_GT(args.hc_eps(), 0.0f)
      << "deepseek_v4 config hc_eps must be > 0, got " << args.hc_eps();
  CHECK_GT(args.factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling.factor/factor, got "
      << args.factor();
  CHECK_GT(args.rope_scaling_attn_factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling_attn_factor, got "
      << args.rope_scaling_attn_factor();
  CHECK_GT(args.rope_theta(), 0.0f)
      << "deepseek_v4 requires positive rope_theta, got " << args.rope_theta();
  CHECK_GT(args.compress_rope_theta(), 0.0f)
      << "deepseek_v4 requires positive compress_rope_theta, got "
      << args.compress_rope_theta();
}

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  constexpr auto preset = DeepseekV4PolicyPreset::kDefault;
  const auto args_policy = build_deepseek_v4_args_policy(preset);
  load_deepseek_v4_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
});

}  // namespace xllm
