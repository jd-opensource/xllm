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

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/hf_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/npu_torch/attention.h"
#include "core/layers/npu_torch/fused_moe.h"
#include "core/util/env_var.h"
#include "core/util/tensor_helper.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

template <typename DecoderLayerType>
class MiniMaxM2ModelImplBase : public torch::nn::Module {
 public:
  explicit MiniMaxM2ModelImplBase(const ModelArgs& args) {
    InterruptionBus::get_instance().subscribe(
        [this](bool interrupted) { layer_forward_interrupted_ = interrupted; });
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  virtual layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 protected:
  int32_t max_seq_len_ = 0;
  std::vector<int64_t> mrope_section_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  std::vector<DecoderLayerType> layers_;
  bool layer_forward_interrupted_ = false;
};

template <typename LlmModelType>
class MiniMaxM2ForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit MiniMaxM2ForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings_ = context.get_model_args().tie_word_embeddings();
    model_ = register_module("model", LlmModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") {
    if (auto* hf_loader = dynamic_cast<HFModelLoader*>(loader.get())) {
      hf_loader->for_each_state_dict([&](const StateDict& state_dict) {
        model_->load_state_dict(state_dict.get_dict_with_prefix(prefix));
        if (tie_word_embeddings_) {
          lm_head_->load_state_dict(
              state_dict.get_dict_with_prefix(prefix + "embed_tokens."));
        } else {
          lm_head_->load_state_dict(
              state_dict.get_dict_with_prefix("lm_head."));
        }
      });
      return;
    }

    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  void update_expert_weight(int32_t layer_id) { (void)layer_id; }

  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  LlmModelType model_{nullptr};
  bool tie_word_embeddings_ = false;
  layer::LmHead lm_head_{nullptr};
};

namespace minimax_m2_detail {

struct DecodePathConfig {
  bool native_decode_attention = true;
  bool native_decode_moe = true;
  int64_t native_decode_moe_min_batch = 8;
};

inline const DecodePathConfig& get_decode_path_config() {
  static const DecodePathConfig config = []() {
    DecodePathConfig cfg;
    cfg.native_decode_attention =
        util::get_bool_env("XLLM_MINIMAX_NATIVE_DECODE_ATTN", true);
    cfg.native_decode_moe =
        util::get_bool_env("XLLM_MINIMAX_NATIVE_DECODE_MOE", true);
    cfg.native_decode_moe_min_batch = std::max<int64_t>(
        1, util::get_int_env("XLLM_MINIMAX_NATIVE_DECODE_MOE_MIN_BATCH", 8));
    return cfg;
  }();
  return config;
}

inline bool should_use_native_decode_moe(int64_t num_tokens) {
  const auto& cfg = get_decode_path_config();
  return cfg.native_decode_moe && num_tokens >= cfg.native_decode_moe_min_batch;
}

struct MoeShadowCompareConfig {
  bool enabled = false;
  int64_t layer = 0;
  bool log_rank0_only = true;
  bool log_internals = false;
  int64_t sample_count = 8;
};

inline const MoeShadowCompareConfig& get_moe_shadow_compare_config() {
  static const MoeShadowCompareConfig config = []() {
    MoeShadowCompareConfig cfg;
    cfg.enabled = util::get_bool_env("XLLM_MINIMAX_COMPARE_MOE", false);
    cfg.layer = util::get_int_env("XLLM_MINIMAX_COMPARE_MOE_LAYER", 0);
    cfg.log_rank0_only =
        util::get_bool_env("XLLM_MINIMAX_COMPARE_MOE_RANK0_ONLY", true);
    cfg.log_internals =
        util::get_bool_env("XLLM_MINIMAX_COMPARE_MOE_LOG_INTERNALS", false);
    cfg.sample_count = std::max<int64_t>(
        1, util::get_int_env("XLLM_MINIMAX_COMPARE_MOE_SAMPLE_COUNT", 8));
    return cfg;
  }();
  return config;
}

inline bool try_begin_moe_shadow_compare(int64_t layer_id,
                                         const ModelInputParams& input_params) {
  const auto& cfg = get_moe_shadow_compare_config();
  if (!cfg.enabled || cfg.layer != layer_id ||
      !input_params.batch_forward_type.is_decode()) {
    return false;
  }

  static std::mutex mu;
  static bool claimed = false;
  std::lock_guard<std::mutex> lock(mu);
  if (claimed) {
    return false;
  }
  claimed = true;
  return true;
}

inline bool should_log_moe_shadow_compare(int64_t tp_rank) {
  const auto& cfg = get_moe_shadow_compare_config();
  return !cfg.log_rank0_only || tp_rank == 0;
}

inline std::string summarize_tensor_head(const torch::Tensor& tensor,
                                         int64_t sample_count) {
  if (!tensor.defined()) {
    return "undefined";
  }
  auto flat = tensor.detach().reshape({-1});
  const int64_t numel = flat.numel();
  const int64_t head = std::min<int64_t>(sample_count, numel);
  auto flat_head =
      flat.slice(/*dim=*/0, /*start=*/0, /*end=*/head).to(torch::kCPU);
  std::ostringstream oss;
  oss << "shape=" << tensor.sizes() << ", head=[";
  for (int64_t i = 0; i < head; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    switch (flat_head.scalar_type()) {
      case torch::kBool:
        oss << (flat_head[i].item<bool>() ? "true" : "false");
        break;
      case torch::kInt:
        oss << flat_head[i].item<int32_t>();
        break;
      case torch::kLong:
        oss << flat_head[i].item<int64_t>();
        break;
      default:
        oss << std::fixed << std::setprecision(6)
            << flat_head[i].item<double>();
        break;
    }
  }
  if (numel > head) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

inline std::string summarize_nonzero_entries(const torch::Tensor& tensor,
                                             int64_t sample_count) {
  if (!tensor.defined()) {
    return "undefined";
  }
  auto flat = tensor.detach().reshape({-1}).to(torch::kCPU);
  const auto nonzero_mask = flat.ne(0);
  const int64_t nonzero_count = nonzero_mask.sum().item<int64_t>();
  auto nonzero_idx = nonzero_mask.nonzero().reshape({-1});
  const int64_t head = std::min<int64_t>(sample_count, nonzero_idx.numel());

  std::ostringstream oss;
  oss << "shape=" << tensor.sizes() << ", nonzero_count=" << nonzero_count
      << ", entries=[";
  for (int64_t i = 0; i < head; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    const int64_t idx = nonzero_idx[i].item<int64_t>();
    oss << idx << ":";
    switch (flat.scalar_type()) {
      case torch::kBool:
        oss << (flat[idx].item<bool>() ? "true" : "false");
        break;
      case torch::kInt:
        oss << flat[idx].item<int32_t>();
        break;
      case torch::kLong:
        oss << flat[idx].item<int64_t>();
        break;
      default:
        oss << std::fixed << std::setprecision(6) << flat[idx].item<double>();
        break;
    }
  }
  if (nonzero_idx.numel() > head) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

inline std::string summarize_diff_stats(const torch::Tensor& lhs,
                                        const torch::Tensor& rhs) {
  auto diff = (lhs.to(torch::kFloat32) - rhs.to(torch::kFloat32)).abs();
  std::ostringstream oss;
  oss << "max_abs=" << diff.max().to(torch::kCPU).item<double>()
      << ", mean_abs=" << diff.mean().to(torch::kCPU).item<double>();
  return oss.str();
}

struct DebugDumpConfig {
  bool enabled = false;
  int64_t layer = -1;
  bool rank0_only = true;
  bool prefill_only = true;
  std::string dump_dir;
};

inline const DebugDumpConfig& get_debug_dump_config() {
  static const DebugDumpConfig config = []() {
    DebugDumpConfig cfg;
    const char* dump_dir = std::getenv("XLLM_MINIMAX_DEBUG_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
      return cfg;
    }

    cfg.enabled = true;
    cfg.dump_dir = dump_dir;
    cfg.layer = util::get_int_env("XLLM_MINIMAX_DEBUG_LAYER", 0);
    cfg.rank0_only = util::get_bool_env("XLLM_MINIMAX_DEBUG_RANK0_ONLY", true);
    cfg.prefill_only =
        util::get_bool_env("XLLM_MINIMAX_DEBUG_PREFILL_ONLY", true);
    return cfg;
  }();
  return config;
}

inline bool try_begin_debug_dump(
    int64_t layer_id,
    int64_t tp_rank,
    const layer::AttentionMetadata& attn_metadata) {
  const auto& cfg = get_debug_dump_config();
  if (!cfg.enabled || cfg.layer != layer_id) {
    return false;
  }
  if (cfg.rank0_only && tp_rank != 0) {
    return false;
  }
  if (cfg.prefill_only &&
      (!attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)) {
    return false;
  }
  if (attn_metadata.is_dummy) {
    return false;
  }

  static std::mutex mu;
  static bool claimed = false;
  std::lock_guard<std::mutex> lock(mu);
  if (claimed) {
    return false;
  }

  std::filesystem::create_directories(cfg.dump_dir);
  claimed = true;
  return true;
}

inline std::filesystem::path get_debug_dump_path(const std::string& file_name) {
  return std::filesystem::path(get_debug_dump_config().dump_dir) / file_name;
}

inline void dump_tensor(const std::string& file_name,
                        const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return;
  }
  save_tensor_as_pickle(tensor.detach().contiguous().to(torch::kCPU),
                        get_debug_dump_path(file_name).string());
}

inline void write_debug_metadata(int64_t layer_id,
                                 int64_t tp_rank,
                                 const layer::AttentionMetadata& attn_metadata,
                                 const torch::Tensor& positions) {
  std::ofstream ofs(get_debug_dump_path("metadata.txt"));
  CHECK(ofs.good()) << "Cannot open MiniMax debug metadata file";
  ofs << "layer=" << layer_id << "\n";
  ofs << "tp_rank=" << tp_rank << "\n";
  ofs << "is_prefill=" << attn_metadata.is_prefill << "\n";
  ofs << "is_chunked_prefill=" << attn_metadata.is_chunked_prefill << "\n";
  ofs << "is_causal=" << attn_metadata.is_causal << "\n";
  ofs << "max_query_len=" << attn_metadata.max_query_len << "\n";
  ofs << "max_seq_len=" << attn_metadata.max_seq_len << "\n";
  ofs << "positions_shape=" << positions.sizes() << "\n";
  if (attn_metadata.q_seq_lens_host.defined()) {
    ofs << "num_sequences=" << attn_metadata.q_seq_lens_host.numel() << "\n";
  }
}

inline std::string remap_layer_weight_name(const std::string& name) {
  std::string mapped_name = name;
  if (absl::StartsWith(mapped_name, "block_sparse_moe.")) {
    mapped_name =
        absl::StrReplaceAll(mapped_name, {{"block_sparse_moe.", "mlp."}});
  }
  if (mapped_name == "mlp.e_score_correction_bias") {
    return "mlp.gate.e_score_correction_bias";
  }
  mapped_name = absl::StrReplaceAll(mapped_name,
                                    {{".w1.", ".gate_proj."},
                                     {".w2.", ".down_proj."},
                                     {".w3.", ".up_proj."}});
  return mapped_name;
}

inline bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e5m2 || dtype == torch::kFloat8_e4m3fn;
}

inline QuantArgs make_runtime_quant_args(const QuantArgs& src) {
  QuantArgs dst = src;
  if (src.quant_method() == "fp8") {
    // MiniMax native NPU path dequantizes fp8 checkpoint tensors to bf16
    // during load_state_dict, so runtime modules should allocate/use the
    // regular dense weight path instead of an fp8 kernel mode.
    dst.quant_method("");
  }
  return dst;
}

inline torch::Tensor dequantize_fp8_block_weight(
    const torch::Tensor& fp8_weight,
    const torch::Tensor& weight_scale_inv,
    const std::array<int64_t, 2>& block_size) {
  CHECK_EQ(fp8_weight.dim(), 2)
      << "Only 2D fp8 weights are supported, got shape " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.dim(), 2)
      << "FP8 weight scale tensor must be 2D, got shape "
      << weight_scale_inv.sizes();

  const int64_t block_n = block_size[0];
  const int64_t block_k = block_size[1];
  const int64_t n = fp8_weight.size(0);
  const int64_t k = fp8_weight.size(1);
  const int64_t n_tiles = (n + block_n - 1) / block_n;
  const int64_t k_tiles = (k + block_k - 1) / block_k;

  CHECK_EQ(weight_scale_inv.size(0), n_tiles)
      << "Unexpected fp8 scale shape " << weight_scale_inv.sizes()
      << " for weight shape " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.size(1), k_tiles)
      << "Unexpected fp8 scale shape " << weight_scale_inv.sizes()
      << " for weight shape " << fp8_weight.sizes();

  if (n % block_n == 0 && k % block_k == 0) {
    // For tile-aligned weights, broadcast the per-block scales in tiled layout
    // instead of materializing a full [n, k] expanded scale tensor.
    auto weight_bf16 = fp8_weight.to(torch::kBFloat16)
                           .reshape({n_tiles, block_n, k_tiles, block_k});
    auto scale_bf16 =
        weight_scale_inv.to(torch::kBFloat16).reshape({n_tiles, 1, k_tiles, 1});
    return (weight_bf16 * scale_bf16).reshape({n, k});
  }

  auto expanded_scale = weight_scale_inv.repeat_interleave(block_n, 0)
                            .repeat_interleave(block_k, 1);
  expanded_scale = expanded_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                       .slice(/*dim=*/1, /*start=*/0, /*end=*/k)
                       .to(torch::kBFloat16);
  return fp8_weight.to(torch::kBFloat16) * expanded_scale;
}

inline torch::Tensor rotate_half(const torch::Tensor& x) {
  auto chunks = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

inline std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_sin) {
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  const auto& cos = chunks[0];
  const auto& sin = chunks[1];
  auto q_embed = (q * cos) + (rotate_half(q) * sin);
  auto k_embed = (k * cos) + (rotate_half(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

inline torch::Tensor repeat_kv(const torch::Tensor& hidden_states,
                               int64_t n_rep) {
  CHECK_EQ(hidden_states.dim(), 4)
      << "repeat_kv expects [batch, kv_heads, seq, head_dim], got "
      << hidden_states.sizes();
  if (n_rep == 1) {
    return hidden_states;
  }
  const auto batch = hidden_states.size(0);
  const auto num_kv_heads = hidden_states.size(1);
  const auto seq_len = hidden_states.size(2);
  const auto head_dim = hidden_states.size(3);
  return hidden_states.unsqueeze(/*dim=*/2)
      .expand({batch, num_kv_heads, n_rep, seq_len, head_dim})
      .reshape({batch, num_kv_heads * n_rep, seq_len, head_dim});
}

inline torch::Tensor build_attention_mask(int64_t q_len,
                                          int64_t kv_len,
                                          int64_t prefix_len,
                                          int64_t sliding_window,
                                          const torch::Device& device) {
  auto pos_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto q_pos = torch::arange(prefix_len, prefix_len + q_len, pos_options)
                   .view({q_len, 1});
  auto k_pos = torch::arange(kv_len, pos_options).view({1, kv_len});
  auto invalid = k_pos > q_pos;
  if (sliding_window > 0) {
    auto min_k_pos = q_pos - (sliding_window - 1);
    invalid = invalid.logical_or(k_pos < min_k_pos);
  }
  auto mask = torch::zeros(
      {1, 1, q_len, kv_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  mask.masked_fill_(invalid.unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0),
                    -9984.0f);
  return mask;
}

inline bool should_limit_topk_by_group(int64_t num_total_experts,
                                       int64_t num_expert_group,
                                       int64_t topk_group) {
  return num_expert_group > 1 && topk_group > 0 &&
         topk_group < num_expert_group &&
         num_total_experts % num_expert_group == 0;
}

inline torch::Tensor mask_scores_by_selected_groups(const torch::Tensor& scores,
                                                    int64_t num_expert_group,
                                                    int64_t topk_group) {
  const int64_t experts_per_group = scores.size(-1) / num_expert_group;
  const int64_t group_score_topk = std::min<int64_t>(2, experts_per_group);
  auto grouped_scores =
      scores.view({scores.size(0), num_expert_group, experts_per_group});
  auto group_topk = std::get<0>(torch::topk(grouped_scores,
                                            group_score_topk,
                                            /*dim=*/-1,
                                            /*largest=*/true,
                                            /*sorted=*/false));
  auto group_scores = group_topk.sum(/*dim=*/-1);
  auto selected_groups = std::get<1>(torch::topk(group_scores,
                                                 topk_group,
                                                 /*dim=*/-1,
                                                 /*largest=*/true,
                                                 /*sorted=*/false));

  auto selected_mask = torch::zeros(group_scores.sizes(),
                                    group_scores.options().dtype(torch::kBool));
  selected_mask.scatter_(
      /*dim=*/1,
      selected_groups,
      torch::ones(selected_groups.sizes(), selected_mask.options()));

  auto expert_mask =
      selected_mask.unsqueeze(-1)
          .expand({scores.size(0), num_expert_group, experts_per_group})
          .reshape_as(scores);
  return scores.masked_fill(expert_mask.logical_not(), 0.0);
}

}  // namespace minimax_m2_detail

class MiniMaxTensorParallelRMSNormImpl : public torch::nn::Module {
 public:
  MiniMaxTensorParallelRMSNormImpl(int64_t local_dim,
                                   int64_t global_dim,
                                   int64_t replica_factor,
                                   double eps,
                                   ProcessGroup* process_group,
                                   const torch::TensorOptions& options)
      : local_dim_(local_dim),
        global_dim_(global_dim),
        replica_factor_(replica_factor),
        eps_(eps),
        process_group_(process_group) {
    CHECK(process_group_ != nullptr)
        << "MiniMaxTensorParallelRMSNorm requires tp process group";
    CHECK_GT(replica_factor_, 0);
    CHECK_EQ(process_group_->world_size() % replica_factor_, 0)
        << "tp world size " << process_group_->world_size()
        << " must be divisible by replica factor " << replica_factor_;
    const int64_t effective_world_size =
        process_group_->world_size() / replica_factor_;
    CHECK_GT(effective_world_size, 0);
    CHECK_EQ(global_dim_ % effective_world_size, 0)
        << "global_dim " << global_dim_
        << " must be divisible by effective world size "
        << effective_world_size;
    CHECK_EQ(local_dim_, global_dim_ / effective_world_size)
        << "unexpected local shard size for TP RMSNorm";

    weight_ = register_parameter(
        "weight", torch::empty({local_dim_}, options), /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& input) {
    auto org_shape = input.sizes().vec();
    auto input_2d = input.reshape({-1, local_dim_});
    auto input_fp32 = input_2d.to(torch::kFloat32);
    auto sq_sum = (input_fp32 * input_fp32).sum(/*dim=*/-1, /*keepdim=*/true);
    if (process_group_->world_size() > 1) {
      sq_sum = parallel_state::reduce(sq_sum, process_group_);
    }

    const float inv_global_dim =
        1.0f / static_cast<float>(global_dim_ * replica_factor_);
    // Match HF MiniMax q/k RMSNorm: compute the normalization factor and
    // apply it in fp32, then cast back to the activation dtype before the
    // learned weight multiply.
    auto inv_rms = torch::rsqrt(sq_sum * inv_global_dim + eps_);
    auto normalized = (input_fp32 * inv_rms).to(input_2d.scalar_type());
    auto output = normalized * weight_.view({1, local_dim_});
    return output.view(org_shape);
  }

  const torch::Tensor& weight() const { return weight_; }
  double eps() const { return eps_; }
  ProcessGroup* process_group() const { return process_group_; }

  void load_state_dict(const StateDict& state_dict) {
    if (weight_is_loaded_) {
      return;
    }

    const int64_t rank = process_group_->rank() / replica_factor_;
    const int64_t world_size = process_group_->world_size() / replica_factor_;
    auto tensor = state_dict.get_sharded_tensor("weight", 0, rank, world_size);
    if (!tensor.defined()) {
      return;
    }

    torch::NoGradGuard no_grad;
    CHECK_EQ(weight_.sizes(), tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << "weight";
    weight_.copy_(tensor);
    weight_is_loaded_ = true;
  }

 private:
  DEFINE_WEIGHT(weight);
  int64_t local_dim_ = 0;
  int64_t global_dim_ = 0;
  int64_t replica_factor_ = 1;
  double eps_ = 1e-6;
  ProcessGroup* process_group_ = nullptr;
};
TORCH_MODULE(MiniMaxTensorParallelRMSNorm);

inline std::tuple<torch::Tensor, torch::Tensor> apply_minimax_tp_qk_rms_norm(
    const torch::Tensor& query,
    const torch::Tensor& key,
    MiniMaxTensorParallelRMSNorm& q_norm,
    MiniMaxTensorParallelRMSNorm& k_norm) {
  auto* process_group = q_norm->process_group();
  if (process_group == nullptr || process_group != k_norm->process_group() ||
      !query.device().is_privateuseone() || !key.device().is_privateuseone()) {
    return std::make_tuple(q_norm->forward(query), k_norm->forward(key));
  }

  const auto q_input = query.is_contiguous() ? query : query.contiguous();
  const auto k_input = key.is_contiguous() ? key : key.contiguous();
  auto [q_local, q_inv_rms] = at_npu::native::custom_ops::npu_rms_norm(
      q_input, q_norm->weight(), q_norm->eps());
  auto [k_local, k_inv_rms] = at_npu::native::custom_ops::npu_rms_norm(
      k_input, k_norm->weight(), k_norm->eps());

  auto normalize_inv_rms = [](const torch::Tensor& inv_rms) {
    auto inv_rms_fp32 = inv_rms.to(torch::kFloat32);
    if (inv_rms_fp32.dim() > 0 && inv_rms_fp32.size(-1) != 1) {
      inv_rms_fp32 = inv_rms_fp32.mean(/*dim=*/-1, /*keepdim=*/true);
    }
    return inv_rms_fp32;
  };

  auto q_local_rstd = normalize_inv_rms(q_inv_rms);
  auto k_local_rstd = normalize_inv_rms(k_inv_rms);
  auto q_local_var =
      (q_local_rstd.reciprocal().pow(2) - q_norm->eps()).clamp_min(0.0);
  auto k_local_var =
      (k_local_rstd.reciprocal().pow(2) - k_norm->eps()).clamp_min(0.0);

  auto qk_var = torch::cat({q_local_var, k_local_var}, /*dim=*/-1);
  if (process_group->world_size() > 1) {
    qk_var = parallel_state::reduce(qk_var, process_group);
    qk_var = qk_var / static_cast<double>(process_group->world_size());
  }

  auto q_global_var = qk_var.slice(/*dim=*/-1, /*start=*/0, /*end=*/1);
  auto k_global_var = qk_var.slice(/*dim=*/-1, /*start=*/1, /*end=*/2);
  auto q_global_rstd = torch::rsqrt(q_global_var + q_norm->eps());
  auto k_global_rstd = torch::rsqrt(k_global_var + k_norm->eps());

  q_local = q_local * (q_global_rstd / q_local_rstd).to(q_local.scalar_type());
  k_local = k_local * (k_global_rstd / k_local_rstd).to(k_local.scalar_type());
  return std::make_tuple(std::move(q_local), std::move(k_local));
}

class MiniMaxM2RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  MiniMaxM2RotaryEmbeddingImpl(int64_t rotary_dim,
                               int64_t max_position_embeddings,
                               int64_t rope_theta,
                               const std::vector<int64_t>& mrope_section,
                               const torch::TensorOptions& options)
      : rotary_dim_(rotary_dim), mrope_section_(mrope_section) {
    const auto inv_freq =
        layer::rotary::compute_inv_freq(rotary_dim, rope_theta, options);
    const auto cos_sin =
        layer::rotary::compute_cos_sin_cache(rotary_dim,
                                             max_position_embeddings,
                                             /*interleaved=*/false,
                                             inv_freq,
                                             options);
    cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& positions) const {
    DCHECK_GE(query.size(-1), rotary_dim_);
    auto query_rotary = query.index({"...", ISlice(0, rotary_dim_)});
    auto query_pass = query.index({"...", ISlice(rotary_dim_, None)});
    auto key_rotary = key.index({"...", ISlice(0, rotary_dim_)});
    auto key_pass = key.index({"...", ISlice(rotary_dim_, None)});

    namespace F = torch::nn::functional;
    auto cos_sin = F::embedding(positions, cos_sin_cache_);
    if (positions.dim() == 2 && !mrope_section_.empty()) {
      auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      auto apply = [this](torch::Tensor x) {
        auto sections = mrope_section_;
        sections.insert(sections.end(), sections.begin(), sections.end());
        auto vec = x.split(sections, /*dim=*/-1);
        std::vector<torch::Tensor> selects;
        selects.reserve(vec.size());
        for (int64_t i = 0; i < static_cast<int64_t>(vec.size()); ++i) {
          selects.push_back(vec[i][i % mrope_section_.size()]);
        }
        return torch::cat(selects, /*dim=*/-1);
      };

      auto cos = apply(chunks[0]);
      auto sin = apply(chunks[1]);
      cos_sin = torch::cat({cos, sin}, /*dim=*/-1);
    }

    cos_sin = cos_sin.unsqueeze(/*dim=*/1);
    std::tie(query_rotary, key_rotary) =
        minimax_m2_detail::apply_rotary_pos_emb(
            query_rotary, key_rotary, cos_sin);
    return std::make_tuple(torch::cat({query_rotary, query_pass}, /*dim=*/-1),
                           torch::cat({key_rotary, key_pass}, /*dim=*/-1));
  }

 private:
  torch::Tensor cos_sin_cache_;
  int64_t rotary_dim_ = 0;
  std::vector<int64_t> mrope_section_;
};
TORCH_MODULE(MiniMaxM2RotaryEmbedding);

class MiniMaxM2AttentionImpl : public torch::nn::Module {
 public:
  explicit MiniMaxM2AttentionImpl(const ModelContext& context) {
    const auto& args = context.get_model_args();
    const auto quant_args =
        minimax_m2_detail::make_runtime_quant_args(context.get_quant_args());
    const auto& parallel_args = context.get_parallel_args();
    const auto& options = context.get_tensor_options();
    const int64_t tp_size = parallel_args.tp_group_->world_size();
    const int64_t total_num_heads = args.n_heads();
    const int64_t total_num_kv_heads =
        args.n_kv_heads().value_or(args.n_heads());

    CHECK(total_num_heads % tp_size == 0);
    num_heads_ = total_num_heads / tp_size;
    if (total_num_kv_heads >= tp_size) {
      CHECK(total_num_kv_heads % tp_size == 0);
      num_kv_heads_ = total_num_kv_heads / tp_size;
      num_kv_head_replicas_ = 1;
    } else {
      CHECK(tp_size % total_num_kv_heads == 0);
      num_kv_heads_ = 1;
      num_kv_head_replicas_ = tp_size / total_num_kv_heads;
    }

    head_dim_ = args.head_dim();
    q_size_ = num_heads_ * head_dim_;
    kv_size_ = num_kv_heads_ * head_dim_;
    CHECK_EQ(num_heads_ % num_kv_heads_, 0)
        << "MiniMax local attention heads must be divisible by local kv heads";
    scaling_ = std::sqrt(1.0f / head_dim_);
    use_qk_norm_ = args.use_qk_norm();
    use_per_layer_qk_norm_ = args.qk_norm_type() == "per_layer";
    sliding_window_ = args.sliding_window();

    qkv_proj_ =
        register_module("qkv_proj",
                        layer::QKVParallelLinear(args.hidden_size(),
                                                 num_heads_,
                                                 num_kv_heads_,
                                                 head_dim_,
                                                 num_kv_head_replicas_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 parallel_args,
                                                 options));

    o_proj_ = register_module(
        "o_proj",
        layer::RowParallelLinear(total_num_heads * head_dim_,
                                 args.hidden_size(),
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 parallel_args.tp_group_,
                                 options));

    if (use_qk_norm_) {
      if (use_per_layer_qk_norm_) {
        q_norm_tp_ = register_module(
            "q_norm",
            MiniMaxTensorParallelRMSNorm(q_size_,
                                         total_num_heads * head_dim_,
                                         /*replica_factor=*/1,
                                         args.rms_norm_eps(),
                                         parallel_args.tp_group_,
                                         options));
        k_norm_tp_ = register_module(
            "k_norm",
            MiniMaxTensorParallelRMSNorm(kv_size_,
                                         total_num_kv_heads * head_dim_,
                                         num_kv_head_replicas_,
                                         args.rms_norm_eps(),
                                         parallel_args.tp_group_,
                                         options));
      } else {
        q_norm_ = register_module(
            "q_norm", layer::RMSNorm(head_dim_, args.rms_norm_eps(), options));
        k_norm_ = register_module(
            "k_norm", layer::RMSNorm(head_dim_, args.rms_norm_eps(), options));
      }
    }

    const int64_t rotary_dim =
        args.rotary_dim() > 0 ? args.rotary_dim() : args.head_dim();
    rotary_emb_ = register_module(
        "rope",
        MiniMaxM2RotaryEmbedding(rotary_dim,
                                 args.max_position_embeddings(),
                                 args.rope_theta(),
                                 args.rope_scaling_mrope_section(),
                                 options));
    attn_ = register_module("attn",
                            layer::Attention(num_heads_,
                                             head_dim_,
                                             scaling_,
                                             num_kv_heads_,
                                             args.sliding_window()));
  }

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache) {
    if (attn_metadata.is_dummy) {
      // Empty DP participants still need to walk the model so MoE collectives
      // stay matched, but they must not enter the native decode-attention
      // kernels with zero-length decode metadata.
      return torch::zeros_like(hidden_states);
    }

    auto qkv = qkv_proj_->forward(hidden_states);
    auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
    auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
    auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

    const int64_t num_tokens = q.size(0);
    if (use_qk_norm_) {
      if (use_per_layer_qk_norm_) {
        std::tie(q, k) =
            apply_minimax_tp_qk_rms_norm(q, k, q_norm_tp_, k_norm_tp_);
      }
    }

    auto q_heads = q.view({num_tokens, num_heads_, head_dim_});
    auto k_heads = k.view({num_tokens, num_kv_heads_, head_dim_});
    auto v_heads = v.view({num_tokens, num_kv_heads_, head_dim_});

    if (use_qk_norm_ && !use_per_layer_qk_norm_) {
      q_heads = std::get<0>(q_norm_->forward(q_heads));
      k_heads = std::get<0>(k_norm_->forward(k_heads));
    }

    std::tie(q_heads, k_heads) =
        rotary_emb_->forward(q_heads, k_heads, positions);
    if (!attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill &&
        minimax_m2_detail::get_decode_path_config().native_decode_attention) {
      auto q_flat = q_heads.reshape({num_tokens, q_size_}).contiguous();
      auto k_flat = k_heads.reshape({num_tokens, kv_size_}).contiguous();
      auto v_flat = v_heads.reshape({num_tokens, kv_size_}).contiguous();
      auto out = std::get<0>(
          attn_->forward(attn_metadata, q_flat, k_flat, v_flat, kv_cache));
      return o_proj_->forward(out);
    }
    auto out_heads = forward_eager_attention(
        q_heads, k_heads, v_heads, attn_metadata, kv_cache);
    return o_proj_->forward(out_heads.view({num_tokens, q_size_}));
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
    if (use_qk_norm_) {
      if (use_per_layer_qk_norm_) {
        q_norm_tp_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
        k_norm_tp_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
      } else {
        q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
        k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
      }
    }
  }

 private:
  void store_kv_cache(const layer::AttentionMetadata& attn_metadata,
                      const torch::Tensor& key_states,
                      const torch::Tensor& value_states,
                      KVCache& kv_cache) const {
    if (!attn_metadata.slot_mapping.defined()) {
      return;
    }

    // Match the normal paged-cache path: slot_mapping is expected to already
    // contain only valid cache slots for active tokens. Avoid .item() checks on
    // device tensors here because MiniMax decode runs under ACL graph capture.
    auto slot_mapping =
        attn_metadata.slot_mapping.reshape({-1}).to(torch::kLong);
    auto k_cache = kv_cache.get_k_cache().view({-1, num_kv_heads_, head_dim_});
    auto v_cache = kv_cache.get_v_cache().view({-1, num_kv_heads_, head_dim_});
    k_cache.index_copy_(/*dim=*/0, slot_mapping, key_states.contiguous());
    v_cache.index_copy_(/*dim=*/0, slot_mapping, value_states.contiguous());
  }

  std::tuple<torch::Tensor, torch::Tensor> materialize_cached_kv(
      const layer::AttentionMetadata& attn_metadata,
      int64_t seq_idx,
      int64_t kv_len,
      KVCache& kv_cache) const {
    CHECK(attn_metadata.block_table.defined())
        << "MiniMax eager attention needs block_table for cached KV paths";

    auto k_cache = kv_cache.get_k_cache();
    auto v_cache = kv_cache.get_v_cache();
    CHECK(v_cache.defined()) << "MiniMax eager attention requires v_cache";

    const int64_t block_size = k_cache.size(1);
    const int64_t num_blocks = (kv_len + block_size - 1) / block_size;
    auto block_ids =
        attn_metadata.block_table.index({seq_idx, ISlice(0, num_blocks)})
            .to(k_cache.device())
            .to(torch::kLong);
    auto k_seq =
        k_cache.index_select(/*dim=*/0, block_ids)
            .reshape({num_blocks * block_size, num_kv_heads_, head_dim_})
            .slice(/*dim=*/0, /*start=*/0, /*end=*/kv_len);
    auto v_seq =
        v_cache.index_select(/*dim=*/0, block_ids)
            .reshape({num_blocks * block_size, num_kv_heads_, head_dim_})
            .slice(/*dim=*/0, /*start=*/0, /*end=*/kv_len);
    return std::make_tuple(k_seq.unsqueeze(/*dim=*/0)
                               .transpose(/*dim0=*/1, /*dim1=*/2)
                               .contiguous(),
                           v_seq.unsqueeze(/*dim=*/0)
                               .transpose(/*dim0=*/1, /*dim1=*/2)
                               .contiguous());
  }

  torch::Tensor forward_eager_attention(
      const torch::Tensor& query_states,
      const torch::Tensor& key_states,
      const torch::Tensor& value_states,
      const layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache) const {
    // Correctness-first MiniMax path: bypass the shared NPU ATB attention
    // kernels until their MiniMax contract is validated against the checkpoint.
    if (attn_metadata.is_dummy) {
      return torch::zeros_like(query_states);
    }

    store_kv_cache(attn_metadata, key_states, value_states, kv_cache);

    CHECK(attn_metadata.q_seq_lens_host.defined())
        << "MiniMax eager attention requires q_seq_lens_host";
    auto output = torch::empty_like(query_states);
    const int64_t kv_repeat = num_heads_ / num_kv_heads_;
    torch::Tensor kv_seq_lens_host;
    if (attn_metadata.kv_seq_lens_host.defined()) {
      kv_seq_lens_host = attn_metadata.kv_seq_lens_host.contiguous();
    }

    const auto q_seq_lens_host = attn_metadata.q_seq_lens_host.contiguous();
    const int64_t num_sequences = q_seq_lens_host.numel();
    int64_t q_start = 0;
    for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      const int64_t q_len = q_seq_lens_host[seq_idx].item<int64_t>();
      const int64_t q_end = q_start + q_len;
      if (q_len == 0) {
        q_start = q_end;
        continue;
      }

      auto q_seq = query_states.slice(/*dim=*/0, q_start, q_end)
                       .unsqueeze(/*dim=*/0)
                       .transpose(/*dim0=*/1, /*dim1=*/2)
                       .contiguous();

      torch::Tensor k_seq;
      torch::Tensor v_seq;
      torch::Tensor attn_mask;
      if (attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill) {
        k_seq = key_states.slice(/*dim=*/0, q_start, q_end)
                    .unsqueeze(/*dim=*/0)
                    .transpose(/*dim0=*/1, /*dim1=*/2)
                    .contiguous();
        v_seq = value_states.slice(/*dim=*/0, q_start, q_end)
                    .unsqueeze(/*dim=*/0)
                    .transpose(/*dim0=*/1, /*dim1=*/2)
                    .contiguous();
        attn_mask = minimax_m2_detail::build_attention_mask(
            q_len, q_len, /*prefix_len=*/0, sliding_window_, q_seq.device());
      } else {
        CHECK(kv_seq_lens_host.defined())
            << "MiniMax eager attention requires kv_seq_lens for cached paths";
        const int64_t kv_len = kv_seq_lens_host[seq_idx].item<int64_t>();
        const int64_t prefix_len = std::max<int64_t>(0, kv_len - q_len);
        std::tie(k_seq, v_seq) =
            materialize_cached_kv(attn_metadata, seq_idx, kv_len, kv_cache);
        attn_mask = minimax_m2_detail::build_attention_mask(
            q_len, kv_len, prefix_len, sliding_window_, q_seq.device());
      }

      auto k_full = minimax_m2_detail::repeat_kv(k_seq, kv_repeat);
      auto v_full = minimax_m2_detail::repeat_kv(v_seq, kv_repeat);
      auto q_attn = q_seq.to(torch::kFloat32);
      auto k_attn = k_full.to(torch::kFloat32);
      auto v_attn = v_full.to(torch::kFloat32);
      auto attn_weights =
          torch::matmul(q_attn, k_attn.transpose(/*dim0=*/-2, /*dim1=*/-1)) *
          scaling_;
      attn_weights = attn_weights + attn_mask;
      attn_weights = torch::softmax(attn_weights, /*dim=*/-1, torch::kFloat32);
      auto attn_output =
          torch::matmul(attn_weights, v_attn).to(query_states.scalar_type());
      output.slice(/*dim=*/0, q_start, q_end)
          .copy_(attn_output.squeeze(/*dim=*/0).transpose(/*dim0=*/0,
                                                          /*dim1=*/1));
      q_start = q_end;
    }

    return output;
  }

  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  int64_t sliding_window_ = -1;
  float scaling_ = 1.0f;
  bool use_qk_norm_ = false;
  bool use_per_layer_qk_norm_ = false;

  layer::QKVParallelLinear qkv_proj_{nullptr};
  layer::RowParallelLinear o_proj_{nullptr};
  layer::RMSNorm q_norm_{nullptr};
  layer::RMSNorm k_norm_{nullptr};
  MiniMaxTensorParallelRMSNorm q_norm_tp_{nullptr};
  MiniMaxTensorParallelRMSNorm k_norm_tp_{nullptr};
  MiniMaxM2RotaryEmbedding rotary_emb_{nullptr};
  layer::Attention attn_{nullptr};
};
TORCH_MODULE(MiniMaxM2Attention);

class MiniMaxM2SparseMoEImpl : public torch::nn::Module {
 public:
  explicit MiniMaxM2SparseMoEImpl(const ModelContext& context,
                                  int64_t layer_id) {
    const auto& args = context.get_model_args();
    const auto& parallel_args = context.get_parallel_args();
    const auto& options = context.get_tensor_options();
    const auto quant_args =
        minimax_m2_detail::make_runtime_quant_args(context.get_quant_args());

    layer_id_ = layer_id;

    if (parallel_args.ep_size() > 1) {
      layer::FusedMoEArgs moe_args;
      ep_moe_ = register_module(
          "ep_moe",
          npu_torch_layer::FusedMoE(
              args, moe_args, quant_args, parallel_args, options));
      return;
    }

    CHECK(parallel_args.tp_group_ != nullptr)
        << "MiniMax sparse MoE requires tp_group";
    CHECK_GT(args.n_routed_experts(), 0);
    CHECK_EQ(args.n_shared_experts(), 0)
        << "MiniMax native correctness-first MoE fallback does not support "
           "shared experts yet";
    CHECK(args.hidden_act() == "silu")
        << "MiniMax native correctness-first MoE fallback only supports "
           "silu activation, got "
        << args.hidden_act();

    tp_group_ = parallel_args.tp_group_;
    rank_ = tp_group_->rank();
    world_size_ = tp_group_->world_size();

    hidden_size_ = args.hidden_size();
    num_experts_ = args.n_routed_experts();
    topk_ = args.num_experts_per_tok();
    num_expert_group_ = std::max<int64_t>(args.n_group(), 1);
    topk_group_ = args.topk_group() > 0 ? args.topk_group()
                                        : std::max<int64_t>(args.n_group(), 1);
    route_scale_ = args.routed_scaling_factor();
    renormalize_ = args.norm_topk_prob();
    hidden_act_ = args.hidden_act();
    scoring_func_ =
        args.scoring_func().empty() ? "softmax" : args.scoring_func();
    local_intermediate_size_ = args.moe_intermediate_size() / world_size_;
    CHECK_EQ(args.moe_intermediate_size() % world_size_, 0)
        << "MiniMax MoE intermediate size must be divisible by TP size";

    gate_weight_ =
        register_parameter("gate_weight",
                           torch::empty({hidden_size_, num_experts_},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias",
        torch::zeros({num_experts_}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // Keep MiniMax MoE weights in a decode-friendly runtime layout so we do
    // not duplicate the full expert weights just to cache transposed copies.
    w13_ = register_parameter(
        "w13",
        torch::empty({num_experts_, hidden_size_, local_intermediate_size_ * 2},
                     options),
        /*requires_grad=*/false);
    w2_ = register_parameter(
        "w2",
        torch::empty({num_experts_, local_intermediate_size_, hidden_size_},
                     options),
        /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params) {
    if (ep_moe_) {
      return ep_moe_->forward(hidden_states, input_params);
    }
    if (input_params.batch_forward_type.is_decode()) {
      if (minimax_m2_detail::try_begin_moe_shadow_compare(layer_id_,
                                                          input_params)) {
        return forward_decode_with_shadow_compare(hidden_states);
      }
      if (minimax_m2_detail::should_use_native_decode_moe(
              hidden_states.size(0))) {
        return forward_native_decode(hidden_states);
      }
      return forward_decode_vectorized_fallback(hidden_states);
    }
    return forward_eager_fallback(hidden_states);
  }

  torch::Tensor forward_eager_fallback(const torch::Tensor& hidden_states) {
    auto hidden_states_2d =
        hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
    auto router_logits =
        torch::matmul(hidden_states_2d.to(torch::kFloat32), gate_weight_);

    torch::Tensor routing_scores;
    if (scoring_func_ == "sigmoid") {
      routing_scores = torch::sigmoid(router_logits.to(torch::kFloat32));
    } else if (scoring_func_ == "softmax") {
      routing_scores =
          torch::softmax(router_logits.to(torch::kFloat32), /*dim=*/-1);
    } else {
      LOG(FATAL) << "Unsupported MiniMax scoring_func " << scoring_func_;
    }

    auto choice_scores = routing_scores;
    if (e_score_correction_bias_.defined()) {
      choice_scores = choice_scores + e_score_correction_bias_;
    }
    if (minimax_m2_detail::should_limit_topk_by_group(
            num_experts_, num_expert_group_, topk_group_)) {
      choice_scores = minimax_m2_detail::mask_scores_by_selected_groups(
          choice_scores, num_expert_group_, topk_group_);
    }

    auto topk_result = torch::topk(choice_scores,
                                   topk_,
                                   /*dim=*/-1,
                                   /*largest=*/true,
                                   /*sorted=*/false);
    auto topk_ids = std::get<1>(topk_result).to(torch::kLong).contiguous();
    auto topk_weights = routing_scores.gather(/*dim=*/1, topk_ids).contiguous();
    if (renormalize_) {
      topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
    }
    if (route_scale_ != 1.0) {
      topk_weights = topk_weights * route_scale_;
    }

    auto output = torch::zeros_like(hidden_states_2d);
    for (int64_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
      auto expert_matches = (topk_ids == expert_idx).nonzero();
      if (expert_matches.numel() == 0) {
        continue;
      }

      auto token_idx = expert_matches.index({ISlice(), 0}).to(torch::kLong);
      auto topk_slot = expert_matches.index({ISlice(), 1}).to(torch::kLong);
      auto current_states = hidden_states_2d.index_select(/*dim=*/0, token_idx);
      auto current_w13 = w13_.index({expert_idx});
      auto current_w2 = w2_.index({expert_idx});

      auto gate_up = torch::matmul(current_states, current_w13);
      auto gate_proj = gate_up.slice(/*dim=*/-1,
                                     /*start=*/0,
                                     /*end=*/local_intermediate_size_);
      auto up_proj = gate_up.slice(/*dim=*/-1,
                                   /*start=*/local_intermediate_size_,
                                   /*end=*/local_intermediate_size_ * 2);
      auto activated = torch::silu(gate_proj) * up_proj;
      auto expert_out = torch::matmul(activated, current_w2);
      auto combine_weight = topk_weights.index({token_idx, topk_slot})
                                .to(expert_out.scalar_type())
                                .unsqueeze(/*dim=*/-1);
      output.index_add_(/*dim=*/0, token_idx, expert_out * combine_weight);
    }

    if (world_size_ > 1) {
      output = parallel_state::reduce(output, tp_group_);
    }
    return output.reshape(hidden_states.sizes());
  }

  torch::Tensor forward_native_decode(const torch::Tensor& hidden_states) {
    return forward_native_decode_impl(prepare_decode_routing(hidden_states),
                                      /*trace=*/nullptr,
                                      hidden_states.sizes());
  }

  torch::Tensor forward_decode_vectorized_fallback(
      const torch::Tensor& hidden_states) {
    return forward_decode_vectorized_fallback_impl(
        prepare_decode_routing(hidden_states),
        /*trace=*/nullptr,
        hidden_states.sizes());
  }

  void load_state_dict(const StateDict& state_dict) {
    if (state_dict.size() == 0) {
      return;
    }
    if (ep_moe_) {
      ep_moe_->load_state_dict(state_dict);
      return;
    }

    auto gate_weight = state_dict.get_tensor("gate.weight");
    if (gate_weight.defined() && !gate_weight_is_loaded_) {
      torch::NoGradGuard no_grad;
      auto gate_weight_t = gate_weight.to(gate_weight_.dtype())
                               .transpose(/*dim0=*/0, /*dim1=*/1);
      CHECK_EQ(gate_weight_.sizes(), gate_weight_t.sizes())
          << "MiniMax sparse MoE gate.weight size mismatch";
      gate_weight_.copy_(gate_weight_t);
      gate_weight_is_loaded_ = true;
    }
    auto bias = state_dict.get_tensor("gate.e_score_correction_bias");
    if (bias.defined() && !e_score_correction_bias_is_loaded_) {
      torch::NoGradGuard no_grad;
      CHECK_EQ(e_score_correction_bias_.sizes(), bias.sizes())
          << "MiniMax e_score_correction_bias size mismatch";
      e_score_correction_bias_.copy_(bias.to(e_score_correction_bias_.dtype()));
      e_score_correction_bias_is_loaded_ = true;
    }

    torch::NoGradGuard no_grad;
    for (int64_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
      const std::string expert_prefix =
          "experts." + std::to_string(expert_idx) + ".";
      auto expert_state = state_dict.get_dict_with_prefix(expert_prefix);
      auto w1 = expert_state.get_sharded_tensor(
          "gate_proj.weight", /*dim=*/0, rank_, world_size_);
      auto w3 = expert_state.get_sharded_tensor(
          "up_proj.weight", /*dim=*/0, rank_, world_size_);
      auto w2 = expert_state.get_sharded_tensor(
          "down_proj.weight", /*dim=*/1, rank_, world_size_);
      auto current_w13 = w13_.index({expert_idx});
      auto current_w2 = w2_.index({expert_idx});
      if (w1.defined()) {
        auto current_w1 = current_w13.slice(/*dim=*/1,
                                            /*start=*/0,
                                            /*end=*/local_intermediate_size_);
        auto w1_t = w1.transpose(/*dim0=*/0, /*dim1=*/1);
        CHECK_EQ(current_w1.sizes(), w1_t.sizes())
            << "MiniMax sparse MoE gate_proj weight size mismatch for expert "
            << expert_idx;
        current_w1.copy_(w1_t);
        w13_is_loaded_ = true;
      }
      if (w3.defined()) {
        auto current_w3 = current_w13.slice(
            /*dim=*/1,
            /*start=*/local_intermediate_size_,
            /*end=*/local_intermediate_size_ * 2);
        auto w3_t = w3.transpose(/*dim0=*/0, /*dim1=*/1);
        CHECK_EQ(current_w3.sizes(), w3_t.sizes())
            << "MiniMax sparse MoE up_proj weight size mismatch for expert "
            << expert_idx;
        current_w3.copy_(w3_t);
        w13_is_loaded_ = true;
      }
      if (w2.defined()) {
        auto w2_t = w2.transpose(/*dim0=*/0, /*dim1=*/1);
        CHECK_EQ(current_w2.sizes(), w2_t.sizes())
            << "MiniMax sparse MoE down_proj weight size mismatch for expert "
            << expert_idx;
        current_w2.copy_(w2_t);
        w2_is_loaded_ = true;
      }
    }
  }

 private:
  struct DecodeRoutingResult {
    torch::Tensor hidden_states_2d;
    torch::Tensor topk_ids_i32;
    torch::Tensor topk_weights;
  };

  struct NativeDecodeTrace {
    torch::Tensor expand_hidden_states;
    torch::Tensor expand_row_ids;
    torch::Tensor group_list;
    torch::Tensor gemm1_out;
    torch::Tensor act_out;
    torch::Tensor gemm2_out;
  };

  struct FallbackDecodeTrace {
    torch::Tensor expert_idx;
    torch::Tensor expanded_hidden_states;
    torch::Tensor gate_up;
    torch::Tensor act_out;
    torch::Tensor expert_out;
  };

  DecodeRoutingResult prepare_decode_routing(
      const torch::Tensor& hidden_states) const {
    auto hidden_states_2d =
        hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
    auto router_logits =
        torch::matmul(hidden_states_2d.to(torch::kFloat32), gate_weight_);

    torch::Tensor routing_scores;
    if (scoring_func_ == "sigmoid") {
      routing_scores = torch::sigmoid(router_logits.to(torch::kFloat32));
    } else if (scoring_func_ == "softmax") {
      routing_scores =
          torch::softmax(router_logits.to(torch::kFloat32), /*dim=*/-1);
    } else {
      LOG(FATAL) << "Unsupported MiniMax scoring_func " << scoring_func_;
    }

    auto choice_scores = routing_scores;
    if (e_score_correction_bias_.defined()) {
      choice_scores = choice_scores + e_score_correction_bias_;
    }
    if (minimax_m2_detail::should_limit_topk_by_group(
            num_experts_, num_expert_group_, topk_group_)) {
      choice_scores = minimax_m2_detail::mask_scores_by_selected_groups(
          choice_scores, num_expert_group_, topk_group_);
    }

    auto topk_result = torch::topk(choice_scores,
                                   topk_,
                                   /*dim=*/-1,
                                   /*largest=*/true,
                                   /*sorted=*/false);
    auto topk_ids_i32 = std::get<1>(topk_result).to(torch::kInt32).contiguous();
    auto topk_weights = routing_scores.gather(
        /*dim=*/1, topk_ids_i32.to(torch::kLong).contiguous());
    if (renormalize_) {
      topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
    }
    if (route_scale_ != 1.0) {
      topk_weights = topk_weights * route_scale_;
    }
    topk_weights = topk_weights.contiguous();

    return {hidden_states_2d, topk_ids_i32, topk_weights};
  }

  torch::Tensor forward_native_decode_impl(const DecodeRoutingResult& routing,
                                           NativeDecodeTrace* trace,
                                           torch::IntArrayRef output_shape) {
    xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
    moe_init_routing_params.x = routing.hidden_states_2d;
    moe_init_routing_params.expert_idx = routing.topk_ids_i32;
    moe_init_routing_params.scale = std::nullopt;
    moe_init_routing_params.offset = std::nullopt;
    moe_init_routing_params.active_num =
        routing.hidden_states_2d.size(0) * topk_;
    moe_init_routing_params.expert_capacity = 0;
    moe_init_routing_params.expert_num = num_experts_;
    moe_init_routing_params.drop_pad_mode = 0;
    moe_init_routing_params.expert_tokens_num_type = 1;
    moe_init_routing_params.expert_tokens_num_flag = true;
    moe_init_routing_params.row_idx_type = 0;
    std::array<int64_t, 2> active_expert_range = {0, num_experts_};
    moe_init_routing_params.active_expert_range = torch::IntArrayRef(
        active_expert_range.data(), active_expert_range.size());
    moe_init_routing_params.quant_mode = -1;
    auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] =
        xllm::kernel::moe_init_routing_v2(moe_init_routing_params);
    (void)dynamic_scale;

    auto run_group_gemm = [&](const torch::Tensor& input,
                              const torch::Tensor& weight) -> torch::Tensor {
      xllm::kernel::GroupGemmParams params;
      params.a = input;
      params.b = weight;
      params.group_list = group_list;
      params.split_item = 2;
      params.group_type = 0;
      params.group_list_type = 1;
      return xllm::kernel::group_gemm(params);
    };

    auto w1_gemm =
        w13_.slice(/*dim=*/2, /*start=*/0, /*end=*/local_intermediate_size_);
    auto gate_proj = run_group_gemm(expand_hidden_states, w1_gemm);

    auto w3_gemm = w13_.slice(/*dim=*/2,
                              /*start=*/local_intermediate_size_,
                              /*end=*/local_intermediate_size_ * 2);
    auto up_proj = run_group_gemm(expand_hidden_states, w3_gemm);

    auto gemm1_out = torch::cat({gate_proj, up_proj}, /*dim=*/1);
    auto act_out = torch::silu(gate_proj) * up_proj;

    xllm::kernel::GroupGemmParams group_gemm2_params;
    group_gemm2_params.a = act_out;
    group_gemm2_params.b = w2_;
    group_gemm2_params.group_list = group_list;
    group_gemm2_params.split_item = 2;
    group_gemm2_params.group_type = 0;
    group_gemm2_params.group_list_type = 1;
    auto gemm2_out = xllm::kernel::group_gemm(group_gemm2_params);

    xllm::kernel::MoeCombineResultParams moe_combine_params;
    moe_combine_params.input = gemm2_out;
    moe_combine_params.reduce_weight = routing.topk_weights;
    moe_combine_params.gather_ids = expand_row_ids;
    auto output = xllm::kernel::moe_combine_result(moe_combine_params);
    if (world_size_ > 1) {
      output = parallel_state::reduce(output, tp_group_);
    }
    if (trace != nullptr) {
      trace->expand_hidden_states = expand_hidden_states;
      trace->expand_row_ids = expand_row_ids;
      trace->group_list = group_list;
      trace->gemm1_out = gemm1_out;
      trace->act_out = act_out;
      trace->gemm2_out = gemm2_out;
    }
    return output.reshape(output_shape);
  }

  torch::Tensor forward_decode_vectorized_fallback_impl(
      const DecodeRoutingResult& routing,
      FallbackDecodeTrace* trace,
      torch::IntArrayRef output_shape) {
    auto topk_ids = routing.topk_ids_i32.to(torch::kLong).contiguous();
    const int64_t num_tokens = routing.hidden_states_2d.size(0);
    auto token_idx =
        torch::arange(num_tokens,
                      routing.hidden_states_2d.options().dtype(torch::kLong))
            .unsqueeze(/*dim=*/1)
            .expand({num_tokens, topk_})
            .reshape({-1})
            .contiguous();
    auto expert_idx = topk_ids.reshape({-1}).contiguous();
    auto combine_weight = routing.topk_weights.reshape({-1})
                              .to(routing.hidden_states_2d.scalar_type())
                              .unsqueeze(/*dim=*/-1)
                              .contiguous();

    auto expanded_hidden_states = routing.hidden_states_2d.unsqueeze(/*dim=*/1)
                                      .expand({num_tokens, topk_, hidden_size_})
                                      .reshape({-1, hidden_size_})
                                      .contiguous();
    auto selected_w13 = w13_.index_select(/*dim=*/0, expert_idx);
    auto gate_up =
        torch::bmm(expanded_hidden_states.unsqueeze(/*dim=*/1), selected_w13)
            .squeeze(/*dim=*/1);
    auto gate_proj = gate_up.slice(/*dim=*/-1,
                                   /*start=*/0,
                                   /*end=*/local_intermediate_size_);
    auto up_proj = gate_up.slice(/*dim=*/-1,
                                 /*start=*/local_intermediate_size_,
                                 /*end=*/local_intermediate_size_ * 2);
    auto activated = torch::silu(gate_proj) * up_proj;

    auto selected_w2 = w2_.index_select(/*dim=*/0, expert_idx);
    auto expert_out = torch::bmm(activated.unsqueeze(/*dim=*/1), selected_w2)
                          .squeeze(/*dim=*/1);

    auto output = torch::zeros_like(routing.hidden_states_2d);
    output.index_add_(/*dim=*/0, token_idx, expert_out * combine_weight);
    if (world_size_ > 1) {
      output = parallel_state::reduce(output, tp_group_);
    }
    if (trace != nullptr) {
      trace->expert_idx = expert_idx;
      trace->expanded_hidden_states = expanded_hidden_states;
      trace->gate_up = gate_up;
      trace->act_out = activated;
      trace->expert_out = expert_out;
    }
    return output.reshape(output_shape);
  }

  torch::Tensor forward_decode_with_shadow_compare(
      const torch::Tensor& hidden_states) {
    auto routing = prepare_decode_routing(hidden_states);
    NativeDecodeTrace native_trace;
    FallbackDecodeTrace fallback_trace;
    auto native_output = forward_native_decode_impl(
        routing, &native_trace, hidden_states.sizes());
    auto fallback_output = forward_decode_vectorized_fallback_impl(
        routing, &fallback_trace, hidden_states.sizes());

    if (minimax_m2_detail::should_log_moe_shadow_compare(rank_)) {
      auto diff = (native_output.to(torch::kFloat32) -
                   fallback_output.to(torch::kFloat32))
                      .abs();
      const double max_abs = diff.max().to(torch::kCPU).item<double>();
      const double mean_abs = diff.mean().to(torch::kCPU).item<double>();
      const auto& cfg = minimax_m2_detail::get_moe_shadow_compare_config();
      auto native_order =
          native_trace.expand_row_ids.to(torch::kLong).contiguous();
      auto fallback_expert_idx_native =
          fallback_trace.expert_idx.index_select(/*dim=*/0, native_order);
      auto fallback_hidden_native =
          fallback_trace.expanded_hidden_states.index_select(/*dim=*/0,
                                                             native_order);
      auto fallback_gemm1_native =
          fallback_trace.gate_up.index_select(/*dim=*/0, native_order);
      auto fallback_act_native =
          fallback_trace.act_out.index_select(/*dim=*/0, native_order);
      auto fallback_gemm2_native =
          fallback_trace.expert_out.index_select(/*dim=*/0, native_order);
      LOG(INFO) << "MiniMax MoE shadow compare at layer " << layer_id_
                << ": max_abs=" << max_abs << ", mean_abs=" << mean_abs
                << ", active_path="
                << (minimax_m2_detail::should_use_native_decode_moe(
                        routing.hidden_states_2d.size(0))
                        ? "native"
                        : "fallback");
      LOG(INFO) << "MiniMax MoE shadow compare topk_ids: "
                << minimax_m2_detail::summarize_tensor_head(
                       routing.topk_ids_i32, cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare topk_weights: "
                << minimax_m2_detail::summarize_tensor_head(
                       routing.topk_weights, cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare expand_row_ids: "
                << minimax_m2_detail::summarize_tensor_head(
                       native_trace.expand_row_ids, cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare group_list(nonzero): "
                << minimax_m2_detail::summarize_nonzero_entries(
                       native_trace.group_list, cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare native-order expert_idx: "
                << minimax_m2_detail::summarize_tensor_head(
                       fallback_expert_idx_native, cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare expand_hidden_states diff: "
                << minimax_m2_detail::summarize_diff_stats(
                       native_trace.expand_hidden_states,
                       fallback_hidden_native);
      LOG(INFO) << "MiniMax MoE shadow compare gemm1_out diff: "
                << minimax_m2_detail::summarize_diff_stats(
                       native_trace.gemm1_out, fallback_gemm1_native);
      LOG(INFO) << "MiniMax MoE shadow compare act_out diff: "
                << minimax_m2_detail::summarize_diff_stats(native_trace.act_out,
                                                           fallback_act_native);
      LOG(INFO) << "MiniMax MoE shadow compare gemm2_out diff: "
                << minimax_m2_detail::summarize_diff_stats(
                       native_trace.gemm2_out, fallback_gemm2_native);
      LOG(INFO) << "MiniMax MoE shadow compare native_output: "
                << minimax_m2_detail::summarize_tensor_head(native_output,
                                                            cfg.sample_count);
      LOG(INFO) << "MiniMax MoE shadow compare fallback_output: "
                << minimax_m2_detail::summarize_tensor_head(fallback_output,
                                                            cfg.sample_count);
      if (cfg.log_internals) {
        LOG(INFO) << "MiniMax MoE shadow compare native expand_hidden_states: "
                  << minimax_m2_detail::summarize_tensor_head(
                         native_trace.expand_hidden_states, cfg.sample_count);
        LOG(INFO) << "MiniMax MoE shadow compare fallback "
                     "expand_hidden_states(native order): "
                  << minimax_m2_detail::summarize_tensor_head(
                         fallback_hidden_native, cfg.sample_count);
        LOG(INFO) << "MiniMax MoE shadow compare native gemm1_out: "
                  << minimax_m2_detail::summarize_tensor_head(
                         native_trace.gemm1_out, cfg.sample_count);
        LOG(INFO)
            << "MiniMax MoE shadow compare fallback gate_up(native order): "
            << minimax_m2_detail::summarize_tensor_head(fallback_gemm1_native,
                                                        cfg.sample_count);
        LOG(INFO) << "MiniMax MoE shadow compare native act_out: "
                  << minimax_m2_detail::summarize_tensor_head(
                         native_trace.act_out, cfg.sample_count);
        LOG(INFO)
            << "MiniMax MoE shadow compare fallback act_out(native order): "
            << minimax_m2_detail::summarize_tensor_head(fallback_act_native,
                                                        cfg.sample_count);
        LOG(INFO) << "MiniMax MoE shadow compare native gemm2_out: "
                  << minimax_m2_detail::summarize_tensor_head(
                         native_trace.gemm2_out, cfg.sample_count);
        LOG(INFO)
            << "MiniMax MoE shadow compare fallback expert_out(native order): "
            << minimax_m2_detail::summarize_tensor_head(fallback_gemm2_native,
                                                        cfg.sample_count);
      }
    }

    if (minimax_m2_detail::should_use_native_decode_moe(
            routing.hidden_states_2d.size(0))) {
      return native_output;
    }
    return fallback_output;
  }

  DEFINE_WEIGHT(gate_weight);
  DEFINE_WEIGHT(e_score_correction_bias);
  DEFINE_WEIGHT(w13);
  DEFINE_WEIGHT(w2);

  int64_t layer_id_ = -1;
  int64_t rank_ = 0;
  int64_t world_size_ = 1;
  int64_t hidden_size_ = 0;
  int64_t num_experts_ = 0;
  int64_t topk_ = 0;
  int64_t num_expert_group_ = 1;
  int64_t topk_group_ = 1;
  int64_t local_intermediate_size_ = 0;
  double route_scale_ = 1.0;
  bool renormalize_ = true;
  std::string hidden_act_;
  std::string scoring_func_;
  ProcessGroup* tp_group_ = nullptr;
  npu_torch_layer::FusedMoE ep_moe_{nullptr};
};
TORCH_MODULE(MiniMaxM2SparseMoE);

class MiniMaxM2DecoderLayerImpl : public torch::nn::Module {
 public:
  MiniMaxM2DecoderLayerImpl(const ModelContext& context, int32_t layer_id) {
    const auto& model_args = context.get_model_args();
    const auto quant_args =
        minimax_m2_detail::make_runtime_quant_args(context.get_quant_args());
    const auto& parallel_args = context.get_parallel_args();
    const auto& options = context.get_tensor_options();

    layer_id_ = layer_id;
    tp_rank_ = parallel_args.tp_group_ != nullptr
                   ? parallel_args.tp_group_->rank()
                   : 0;
    attention_ = register_module("self_attn", MiniMaxM2Attention(context));
    input_norm_ = register_module(
        "input_layernorm",
        layer::RMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));
    post_norm_ = register_module(
        "post_attention_layernorm",
        layer::RMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));

    if (model_args.n_routed_experts() > 0) {
      moe_mlp_ = register_module("mlp", MiniMaxM2SparseMoE(context, layer_id));
    } else {
      dense_mlp_ =
          register_module("mlp",
                          layer::DenseMLP(model_args.hidden_size(),
                                          model_args.intermediate_size(),
                                          /*is_gated=*/true,
                                          /*has_bias=*/false,
                                          model_args.hidden_act(),
                                          /*enable_result_reduction=*/true,
                                          quant_args,
                                          parallel_args.tp_group_,
                                          options));
    }
  }

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    if (x.numel() == 0) {
      // Idle DP decode participants must still join the routed-MoE collectives,
      // but running norm/attention on an empty token axis is unnecessary and
      // brittle. Use the empty hidden state itself as the source of truth here
      // so the fast path still holds even if upstream dummy metadata drifts.
      if (moe_mlp_) {
        return moe_mlp_->forward(x, input_params);
      }
      return x;
    }

    bool debug_dump_active = minimax_m2_detail::try_begin_debug_dump(
        layer_id_, tp_rank_, attn_metadata);
    if (debug_dump_active) {
      auto hidden_in = residual.has_value() ? (residual.value() + x) : x;
      LOG(INFO) << "MiniMax debug dump activated for layer " << layer_id_
                << " at "
                << minimax_m2_detail::get_debug_dump_config().dump_dir;
      minimax_m2_detail::write_debug_metadata(
          layer_id_, tp_rank_, attn_metadata, positions);
      minimax_m2_detail::dump_tensor("positions.pt", positions);
      minimax_m2_detail::dump_tensor("q_seq_lens_host.pt",
                                     attn_metadata.q_seq_lens_host);
      minimax_m2_detail::dump_tensor("kv_seq_lens_host.pt",
                                     attn_metadata.kv_seq_lens_host);
      minimax_m2_detail::dump_tensor("block_table.pt",
                                     attn_metadata.block_table);
      minimax_m2_detail::dump_tensor("slot_mapping.pt",
                                     attn_metadata.slot_mapping);
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_hidden_in.pt", hidden_in);
    }

    if (!residual.has_value()) {
      residual = x;
      x = std::get<0>(input_norm_->forward(x));
    } else {
      std::tie(x, residual) = input_norm_->forward(x, residual);
    }
    if (debug_dump_active) {
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_input_norm_out.pt", x);
    }

    x = attention_->forward(positions, x, attn_metadata, kv_cache);
    if (debug_dump_active) {
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_attn_out.pt", x);
    }
    std::tie(x, residual) = post_norm_->forward(x, residual);
    if (debug_dump_active) {
      minimax_m2_detail::dump_tensor("layer_" + std::to_string(layer_id_) +
                                         "_hidden_after_attn_residual.pt",
                                     residual.value());
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_post_norm_out.pt", x);
    }
    if (moe_mlp_) {
      x = moe_mlp_->forward(x, input_params);
    } else {
      x = dense_mlp_->forward(x);
    }
    if (debug_dump_active) {
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_moe_out.pt", x);
      minimax_m2_detail::dump_tensor(
          "layer_" + std::to_string(layer_id_) + "_hidden_out.pt",
          residual.value() + x);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    input_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("input_layernorm."));
    post_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("post_attention_layernorm."));
    if (moe_mlp_) {
      moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
    } else {
      dense_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
    }
  }

 private:
  int32_t layer_id_ = -1;
  int64_t tp_rank_ = 0;
  MiniMaxM2Attention attention_{nullptr};
  layer::DenseMLP dense_mlp_{nullptr};
  MiniMaxM2SparseMoE moe_mlp_{nullptr};
  layer::RMSNorm input_norm_{nullptr};
  layer::RMSNorm post_norm_{nullptr};
};
TORCH_MODULE(MiniMaxM2DecoderLayer);

class MiniMaxM2ModelImpl
    : public MiniMaxM2ModelImplBase<MiniMaxM2DecoderLayer> {
 public:
  explicit MiniMaxM2ModelImpl(const ModelContext& context)
      : MiniMaxM2ModelImplBase<MiniMaxM2DecoderLayer>(
            context.get_model_args()) {
    const auto& options = context.get_tensor_options();
    const auto& model_args = context.get_model_args();
    const auto& quant_args = context.get_quant_args();
    const auto& parallel_args = context.get_parallel_args();

    if (quant_args.quant_method() == "fp8") {
      enable_fp8_dynamic_dequant_ = true;
      if (quant_args.weight_block_size().size() == 2 &&
          quant_args.weight_block_size()[0] > 0 &&
          quant_args.weight_block_size()[1] > 0) {
        fp8_weight_block_size_ = {quant_args.weight_block_size()[0],
                                  quant_args.weight_block_size()[1]};
      }
    }

    layers_.reserve(model_args.n_layers());
    hidden_size_ = model_args.hidden_size();
    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             parallel_args,
                                             options));
    norm_ = register_module("norm", layer::RMSNorm(context));
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      layers_.push_back(MiniMaxM2DecoderLayer(context, i));
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    torch::Tensor h;
    if (input_params.input_embedding.defined()) {
      h = input_params.input_embedding;
    } else if (tokens.numel() == 0) {
      h = torch::empty({0, hidden_size_}, embed_tokens_->weight().options());
    } else {
      h = embed_tokens_(tokens);
    }

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(modified_input_params, h));
    }

    auto& attn_metadata = *(modified_input_params.attn_metadata);
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params);
    }

    if (h.numel() == 0) {
      return ModelOutput(h);
    }

    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));

    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto layer_state_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      std::unordered_map<std::string, torch::Tensor> remapped_layer_state_dict;
      remapped_layer_state_dict.reserve(layer_state_dict.size());
      std::unordered_map<std::string, torch::Tensor> pending_fp8_weights;
      std::unordered_map<std::string, torch::Tensor> pending_fp8_scales;

      for (const auto& [name, tensor] : layer_state_dict) {
        std::string mapped_name =
            minimax_m2_detail::remap_layer_weight_name(name);
        if (enable_fp8_dynamic_dequant_) {
          if (absl::EndsWith(mapped_name, ".weight_scale_inv")) {
            const std::string paired_weight_name =
                mapped_name.substr(0, mapped_name.size() - 10);
            auto pending_weight = pending_fp8_weights.find(paired_weight_name);
            if (pending_weight != pending_fp8_weights.end()) {
              remapped_layer_state_dict.emplace(
                  paired_weight_name,
                  minimax_m2_detail::dequantize_fp8_block_weight(
                      pending_weight->second, tensor, fp8_weight_block_size_));
              pending_fp8_weights.erase(pending_weight);
            } else {
              pending_fp8_scales.emplace(mapped_name, tensor);
            }
            continue;
          }

          if (absl::EndsWith(mapped_name, ".weight") &&
              minimax_m2_detail::is_fp8_dtype(tensor.scalar_type())) {
            const std::string scale_name = mapped_name + "_scale_inv";
            auto pending_scale = pending_fp8_scales.find(scale_name);
            if (pending_scale != pending_fp8_scales.end()) {
              remapped_layer_state_dict.emplace(
                  mapped_name,
                  minimax_m2_detail::dequantize_fp8_block_weight(
                      tensor, pending_scale->second, fp8_weight_block_size_));
              pending_fp8_scales.erase(pending_scale);
            } else {
              pending_fp8_weights.emplace(mapped_name, tensor);
            }
            continue;
          }
        }

        remapped_layer_state_dict.emplace(mapped_name, tensor);
      }

      if (enable_fp8_dynamic_dequant_) {
        CHECK(pending_fp8_weights.empty() && pending_fp8_scales.empty())
            << "Unpaired fp8 MiniMax-M2 weight/scale tensors detected: "
            << "pending_weights=" << pending_fp8_weights.size()
            << ", pending_scales=" << pending_fp8_scales.size();
      }

      layers_[i]->load_state_dict(
          StateDict(std::move(remapped_layer_state_dict)));
    }

    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

 private:
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) {
    if (params.q_max_seq_len == 0) {
      return layer::AttentionMetadataBuilder::build(params);
    }

    max_seq_len_ = std::max(params.kv_max_seq_len, max_seq_len_);
    torch::Tensor attn_mask;
    if (FLAGS_enable_chunked_prefill) {
      const int32_t max_kv_seq = params.kv_max_seq_len;
      const int32_t num_sequences = params.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);
        for (int32_t j = 0; j < num_sequences; ++j) {
          req_mask_vec.emplace_back(
              attn_mask_.gen_append_mask(params.q_seq_lens_vec[j],
                                         params.kv_seq_lens_vec[j],
                                         max_kv_seq,
                                         h.dtype().toScalarType(),
                                         h.device()));
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, h.dtype().toScalarType(), h.device());
      }
    } else {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, h.dtype().toScalarType(), h.device());
    }
    return layer::AttentionMetadataBuilder::build(params, attn_mask);
  }

  int64_t hidden_size_ = 0;
  layer::AttentionMask attn_mask_;
  bool enable_fp8_dynamic_dequant_ = false;
  std::array<int64_t, 2> fp8_weight_block_size_ = {128, 128};
};
TORCH_MODULE(MiniMaxM2Model);

class MiniMaxM2ForCausalLMImpl
    : public MiniMaxM2ForCausalLMImplBase<MiniMaxM2Model> {
 public:
  explicit MiniMaxM2ForCausalLMImpl(const ModelContext& context)
      : MiniMaxM2ForCausalLMImplBase<MiniMaxM2Model>(context) {}
};
TORCH_MODULE(MiniMaxM2ForCausalLM);

REGISTER_CAUSAL_MODEL(minimax_m2, MiniMaxM2ForCausalLM);

REGISTER_MODEL_ARGS(minimax_m2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "minimax_m2");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 200019);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 200020);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 64);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3072);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 1536);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 196608);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 62);
  LOAD_ARG_OR(moe_intermediate_size, "intermediate_size", 1536);
  SET_ARG(n_shared_experts, 0);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(num_experts, "num_local_experts", 256);
  LOAD_ARG_OR(n_routed_experts, "num_local_experts", 256);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 62);
  if (json.contains("attn_type_list")) {
    const auto attn_type_list =
        json.value_or<std::vector<int>>("attn_type_list", std::vector<int>());
    if (!attn_type_list.empty() &&
        args->n_layers() != static_cast<int32_t>(attn_type_list.size())) {
      LOG(WARNING) << "MiniMax config mismatch: num_hidden_layers="
                   << args->n_layers()
                   << ", attn_type_list size=" << attn_type_list.size()
                   << ". Using attn_type_list size.";
      args->n_layers() = static_cast<int32_t>(attn_type_list.size());
    }
  }
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(qk_norm_type, "qk_norm_type", "");
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 5000000.0f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 200064);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  SET_ARG(topk_method, "noaux_tc");
});

}  // namespace xllm
