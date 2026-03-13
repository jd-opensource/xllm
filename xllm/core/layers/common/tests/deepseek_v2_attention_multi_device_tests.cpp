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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "layers/mlu/deepseek_v32_sp_context.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {
namespace test {

namespace {

constexpr int32_t EXIT_CODE_SKIP = 77;

class ScopedBoolFlagValue {
 public:
  ScopedBoolFlagValue(bool& flag, bool value) : flag_(flag), old_(flag) {
    flag_ = value;
  }

  ~ScopedBoolFlagValue() { flag_ = old_; }

 private:
  bool& flag_;
  bool old_;
};

std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    false,
                                    host,
                                    "attention_multi_device_test_group",
                                    device);
}

void check_tensors_close(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol,
                         double atol,
                         const std::string& label) {
  CHECK_EQ(actual.sizes(), expected.sizes())
      << label << " shape mismatch: actual=" << actual.sizes()
      << ", expected=" << expected.sizes();
  torch::Tensor actual_fp32 = actual.to(torch::kFloat32);
  torch::Tensor expected_fp32 = expected.to(torch::kFloat32);
  torch::Tensor diff = torch::abs(actual_fp32 - expected_fp32);
  CHECK(torch::allclose(actual_fp32, expected_fp32, rtol, atol))
      << label << " mismatch. max_diff=" << torch::max(diff).item<float>()
      << ", mean_diff=" << torch::mean(diff).item<float>();
}

ModelArgs create_attention_model_args(int64_t q_lora_rank = 0,
                                      bool enable_indexer = false) {
  ModelArgs args;
  args.model_type() = "deepseek_v32";
  args.hidden_size() = 256;
  args.n_heads() = 4;
  args.max_position_embeddings() = 128;
  args.rope_theta() = 10000.0f;
  args.rms_norm_eps() = 1e-6f;
  args.q_lora_rank() = q_lora_rank;
  args.kv_lora_rank() = enable_indexer ? 256 : 128;
  args.qk_nope_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 128;
  args.index_n_heads() = enable_indexer ? 64 : 0;
  args.index_head_dim() = enable_indexer ? 128 : 0;
  args.index_topk() = enable_indexer ? 8 : 0;
  args.sliding_window() = 0;
  args.rope_scaling_rope_type() = "";
  args.rope_scaling_original_max_position_embeddings() = 128;
  args.rope_scaling_factor() = 1.0f;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_attn_factor() = 1.0f;
  args.rope_scaling_beta_fast() = 1;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  return args;
}

ModelArgs create_glm5_attention_model_args() {
  ModelArgs args =
      create_attention_model_args(/*q_lora_rank=*/64, /*enable_indexer=*/true);
  args.model_type() = "glm_moe_dsa";
  args.qk_nope_head_dim() = 192;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.rope_theta() = 1000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  return args;
}

StateDict create_attention_state_dict(const ModelArgs& args,
                                      const torch::TensorOptions& options) {
  const int64_t hidden_size = args.hidden_size();
  const int64_t num_heads = args.n_heads();
  const int64_t kv_lora_rank = args.kv_lora_rank();
  const int64_t qk_nope_head_dim = args.qk_nope_head_dim();
  const int64_t qk_rope_head_dim = args.qk_rope_head_dim();
  const int64_t v_head_dim = args.v_head_dim();
  const int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  const int64_t q_lora_rank = args.q_lora_rank();
  const int64_t index_n_heads = args.index_n_heads();
  const int64_t index_head_dim = args.index_head_dim();

  std::unordered_map<std::string, torch::Tensor> weights;
  if (q_lora_rank > 0) {
    weights["q_a_proj.weight"] =
        seeded_tensor("attention_multi_device/q_a_proj/weight",
                      {q_lora_rank, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["q_a_layernorm.weight"] =
        seeded_tensor("attention_multi_device/q_a_layernorm/weight",
                      {q_lora_rank},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["q_b_proj.qweight"] =
        seeded_tensor("attention_multi_device/q_b_proj/qweight",
                      {num_heads * qk_head_dim, q_lora_rank},
                      torch::kInt8,
                      options.device());
    weights["q_b_proj.per_channel_scale"] =
        seeded_tensor("attention_multi_device/q_b_proj/per_channel_scale",
                      {num_heads * qk_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
    weights["q_b_proj.smooth"] =
        seeded_tensor("attention_multi_device/q_b_proj/smooth",
                      {q_lora_rank},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
  } else {
    weights["q_proj.qweight"] =
        seeded_tensor("attention_multi_device/q_proj/qweight",
                      {num_heads * qk_head_dim, hidden_size},
                      torch::kInt8,
                      options.device());
    weights["q_proj.per_channel_scale"] =
        seeded_tensor("attention_multi_device/q_proj/per_channel_scale",
                      {num_heads * qk_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
    weights["q_proj.smooth"] =
        seeded_tensor("attention_multi_device/q_proj/smooth",
                      {hidden_size},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
  }

  weights["kv_a_proj_with_mqa.weight"] =
      seeded_tensor("attention_multi_device/kv_a_proj_with_mqa/weight",
                    {kv_lora_rank + qk_rope_head_dim, hidden_size},
                    torch::kBFloat16,
                    options.device());
  weights["kv_a_layernorm.weight"] =
      seeded_tensor("attention_multi_device/kv_a_layernorm/weight",
                    {kv_lora_rank},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.5f);
  weights["kv_b_proj.weight"] =
      seeded_tensor("attention_multi_device/kv_b_proj/weight",
                    {num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank},
                    torch::kBFloat16,
                    options.device());

  weights["o_proj.qweight"] =
      seeded_tensor("attention_multi_device/o_proj/qweight",
                    {hidden_size, num_heads * v_head_dim},
                    torch::kInt8,
                    options.device());
  weights["o_proj.per_channel_scale"] =
      seeded_tensor("attention_multi_device/o_proj/per_channel_scale",
                    {hidden_size},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.25f);
  weights["o_proj.smooth"] =
      seeded_tensor("attention_multi_device/o_proj/smooth",
                    {num_heads * v_head_dim},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.25f);

  if (index_n_heads > 0) {
    weights["indexer.k_norm.bias"] =
        seeded_tensor("attention_multi_device/indexer/k_norm/bias",
                      {index_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.k_norm.weight"] =
        seeded_tensor("attention_multi_device/indexer/k_norm/weight",
                      {index_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.weights_proj.weight"] =
        seeded_tensor("attention_multi_device/indexer/weights_proj/weight",
                      {index_n_heads, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["indexer.wk.weight"] =
        seeded_tensor("attention_multi_device/indexer/wk/weight",
                      {index_head_dim, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["indexer.wq_b.weight"] =
        seeded_tensor("attention_multi_device/indexer/wq_b/weight",
                      {index_n_heads * index_head_dim, q_lora_rank},
                      torch::kBFloat16,
                      options.device());
  }

  return StateDict(std::move(weights));
}

AttentionMetadata create_decode_metadata(const torch::TensorOptions& options) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, 1}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, 0}, int_options);
  metadata.kv_seq_lens = torch::tensor({1}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::tensor({1}, int_options);
  metadata.max_query_len = 1;
  metadata.max_seq_len = 1;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_prefill_metadata(const torch::TensorOptions& options,
                                          int32_t seq_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({seq_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::arange(seq_len, int_options);
  metadata.max_query_len = seq_len;
  metadata.max_seq_len = seq_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_chunked_metadata(const torch::TensorOptions& options,
                                          int32_t prefix_len,
                                          int32_t chunk_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  const int32_t ctx_len = prefix_len + chunk_len;
  metadata.q_cu_seq_lens = torch::tensor({0, chunk_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, ctx_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({ctx_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::arange(prefix_len, ctx_len, int_options);
  metadata.max_query_len = chunk_len;
  metadata.max_seq_len = ctx_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = true;
  metadata.is_dummy = false;
  return metadata;
}

KVCache create_decode_kv_cache(const ModelArgs& args,
                               const torch::TensorOptions& options) {
  const int64_t cache_width = args.qk_rope_head_dim() + args.kv_lora_rank();
  torch::Tensor k_cache = seeded_tensor("attention_multi_device/k_cache",
                                        {8, 1, FLAGS_block_size, cache_width},
                                        torch::kBFloat16,
                                        options.device());
  torch::Tensor index_cache = torch::Tensor();
  if (args.index_head_dim() > 0) {
    index_cache = seeded_tensor("attention_multi_device/index_cache",
                                {8, 1, FLAGS_block_size, args.index_head_dim()},
                                torch::kBFloat16,
                                options.device());
  }
  return KVCache(k_cache, torch::Tensor(), index_cache);
}

torch::Tensor run_attention_decode_once(const ModelArgs& args,
                                        const QuantArgs& quant_args,
                                        const ParallelArgs& parallel_args,
                                        const torch::TensorOptions& options,
                                        const StateDict& state_dict,
                                        const torch::Tensor& positions,
                                        const torch::Tensor& hidden_states,
                                        KVCache& kv_cache,
                                        bool enable_full_weight_path,
                                        bool enable_fused_mla_kernel) {
  ScopedBoolFlagValue flag_guard(FLAGS_enable_prefill_sp,
                                 enable_full_weight_path);
  OptimizationConfig optimization_config;
  optimization_config.enable_fused_mla_kernel = enable_fused_mla_kernel;
  optimization_config.enable_fused_indexer_qk = false;
  DeepseekV2Attention attention(
      args, quant_args, parallel_args, options, optimization_config);
  attention->load_state_dict(state_dict);
  AttentionMetadata metadata = create_decode_metadata(options);
  return attention->forward(positions, hidden_states, metadata, kv_cache);
}

std::tuple<torch::Tensor, torch::Tensor> run_attention_prefill_once(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const StateDict& state_dict,
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    KVCache& kv_cache,
    bool enable_full_weight_path,
    bool enable_fused_mla_kernel,
    BatchForwardType batch_forward_type = BatchForwardType::PREFILL,
    int32_t prefix_len = 0) {
  ScopedBoolFlagValue flag_guard(FLAGS_enable_prefill_sp,
                                 enable_full_weight_path);
  OptimizationConfig optimization_config;
  optimization_config.enable_fused_mla_kernel = enable_fused_mla_kernel;
  optimization_config.enable_fused_indexer_qk = false;
  DeepseekV2Attention attention(
      args, quant_args, parallel_args, options, optimization_config);
  attention->load_state_dict(state_dict);
  const int32_t token_num = static_cast<int32_t>(tokens.numel());
  AttentionMetadata metadata =
      batch_forward_type.is_chunked_prefill()
          ? create_chunked_metadata(options, prefix_len, token_num)
          : create_prefill_metadata(options, token_num);
  std::optional<layer::v32_sp::DeepseekV32SPContext> sp_ctx;
  torch::Tensor local_positions = positions;
  torch::Tensor local_hidden_states = hidden_states;
  if (enable_full_weight_path && args.index_n_heads() > 0) {
    ProcessGroup* sp_group = parallel_args.sp_group_ != nullptr
                                 ? parallel_args.sp_group_
                                 : parallel_args.process_group_;
    sp_ctx = layer::v32_sp::build_deepseek_v32_sp_context(
        metadata,
        batch_forward_type,
        tokens,
        sp_group,
        parallel_args.rank(),
        parallel_args.world_size());
    if (sp_ctx.has_value()) {
      local_positions =
          layer::v32_sp::reorder_to_local_shard(positions, sp_ctx.value());
      local_hidden_states =
          layer::v32_sp::reorder_to_local_shard(hidden_states, sp_ctx.value());
    }
  }
  torch::Tensor output = attention->forward(local_positions,
                                            local_hidden_states,
                                            metadata,
                                            kv_cache,
                                            sp_ctx ? &sp_ctx.value() : nullptr);
  if (sp_ctx.has_value()) {
    output = layer::v32_sp::restore_gathered_to_global_order(
        layer::v32_sp::all_gather_across_ranks(output, sp_ctx.value()),
        sp_ctx.value(),
        layer::v32_sp::GatheredTensorLayout::kPacked);
  }
  return {output, kv_cache.get_k_cache().clone()};
}

int32_t run_attention_test_child(int32_t rank,
                                 int32_t world_size,
                                 int32_t port,
                                 const std::string& host,
                                 bool enable_fused_mla_kernel,
                                 int64_t q_lora_rank) {
  try {
    const int32_t dev_count = xllm::Device::device_count();
    if (dev_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    FLAGS_enable_mla = true;
    FLAGS_block_size = 16;
    const int32_t device_index = rank % dev_count;
    xllm::Device xllm_device(device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();
    auto process_group =
        create_test_process_group(rank, world_size, port, host, device);
    CHECK(process_group) << "Rank " << rank
                         << ": failed to create process group";

    ParallelArgs parallel_args(rank, world_size, process_group.get());
    parallel_args.tp_group_ = process_group.get();
    parallel_args.sp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);
    ModelArgs model_args = create_attention_model_args(q_lora_rank);
    QuantArgs quant_args = create_default_quant_args();
    StateDict state_dict = create_attention_state_dict(model_args, options);

    torch::Tensor hidden_states =
        seeded_tensor("attention_multi_device/hidden_states",
                      {1, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor positions =
        torch::tensor({0}, options.dtype(torch::kInt32).device(device));

    KVCache sharded_kv_cache = create_decode_kv_cache(model_args, options);
    KVCache full_weight_kv_cache = create_decode_kv_cache(model_args, options);

    torch::Tensor sharded_output =
        run_attention_decode_once(model_args,
                                  quant_args,
                                  parallel_args,
                                  options,
                                  state_dict,
                                  positions,
                                  hidden_states,
                                  sharded_kv_cache,
                                  false,
                                  enable_fused_mla_kernel);
    torch::Tensor full_weight_output =
        run_attention_decode_once(model_args,
                                  quant_args,
                                  parallel_args,
                                  options,
                                  state_dict,
                                  positions,
                                  hidden_states,
                                  full_weight_kv_cache,
                                  true,
                                  enable_fused_mla_kernel);

    xllm_device.synchronize_default_stream();
    check_tensors_close(sharded_output,
                        full_weight_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention decode");
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

int32_t run_attention_prefill_test_child(int32_t rank,
                                         int32_t world_size,
                                         int32_t port,
                                         const std::string& host,
                                         bool use_glm5_args = false) {
  try {
    const int32_t dev_count = xllm::Device::device_count();
    if (dev_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    FLAGS_enable_mla = true;
    FLAGS_block_size = 16;
    const int32_t device_index = rank % dev_count;
    xllm::Device xllm_device(device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();
    auto process_group =
        create_test_process_group(rank, world_size, port, host, device);
    CHECK(process_group) << "Rank " << rank
                         << ": failed to create process group";

    ParallelArgs parallel_args(rank, world_size, process_group.get());
    parallel_args.tp_group_ = process_group.get();
    parallel_args.sp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);
    ModelArgs model_args = use_glm5_args ? create_glm5_attention_model_args()
                                         : create_attention_model_args();
    QuantArgs quant_args = create_default_quant_args();
    StateDict state_dict = create_attention_state_dict(model_args, options);

    constexpr int32_t seq_len = 4;
    torch::Tensor tokens =
        torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
    torch::Tensor hidden_states =
        seeded_tensor("attention_multi_device/prefill_hidden_states",
                      {seq_len, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor positions =
        torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

    KVCache sharded_kv_cache = create_decode_kv_cache(model_args, options);
    sharded_kv_cache.get_k_cache().zero_();
    KVCache full_weight_kv_cache = create_decode_kv_cache(model_args, options);
    full_weight_kv_cache.get_k_cache().zero_();

    auto [sharded_output, sharded_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   sharded_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false);
    auto [full_weight_output, full_weight_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   full_weight_kv_cache,
                                   true,
                                   /*enable_fused_mla_kernel=*/false);

    xllm_device.synchronize_default_stream();
    check_tensors_close(sharded_output,
                        full_weight_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention prefill output");
    check_tensors_close(sharded_k_cache,
                        full_weight_k_cache,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention prefill k_cache");
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

int32_t run_attention_prefill_fallback_test_child(int32_t rank,
                                                  int32_t world_size,
                                                  int32_t port,
                                                  const std::string& host,
                                                  bool use_glm5_args = false) {
  try {
    const int32_t dev_count = xllm::Device::device_count();
    if (dev_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    FLAGS_enable_mla = true;
    FLAGS_block_size = 16;
    const int32_t device_index = rank % dev_count;
    xllm::Device xllm_device(device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();
    auto process_group =
        create_test_process_group(rank, world_size, port, host, device);
    CHECK(process_group) << "Rank " << rank
                         << ": failed to create process group";

    ParallelArgs parallel_args(rank, world_size, process_group.get());
    parallel_args.tp_group_ = process_group.get();
    parallel_args.sp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);
    ModelArgs model_args = use_glm5_args ? create_glm5_attention_model_args()
                                         : create_attention_model_args();
    QuantArgs quant_args = create_default_quant_args();
    StateDict state_dict = create_attention_state_dict(model_args, options);

    constexpr int32_t seq_len = 1;
    torch::Tensor tokens =
        torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
    torch::Tensor hidden_states =
        seeded_tensor("attention_multi_device/prefill_hidden_states_fallback",
                      {seq_len, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor positions =
        torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

    KVCache sharded_kv_cache = create_decode_kv_cache(model_args, options);
    sharded_kv_cache.get_k_cache().zero_();
    KVCache full_weight_kv_cache = create_decode_kv_cache(model_args, options);
    full_weight_kv_cache.get_k_cache().zero_();

    auto [sharded_output, sharded_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   sharded_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false);
    auto [full_weight_output, full_weight_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   full_weight_kv_cache,
                                   true,
                                   /*enable_fused_mla_kernel=*/false);

    xllm_device.synchronize_default_stream();
    check_tensors_close(sharded_output,
                        full_weight_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention prefill fallback output");
    check_tensors_close(sharded_k_cache,
                        full_weight_k_cache,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention prefill fallback k_cache");
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

int32_t run_attention_chunked_test_child(int32_t rank,
                                         int32_t world_size,
                                         int32_t port,
                                         const std::string& host) {
  try {
    const int32_t dev_count = xllm::Device::device_count();
    if (dev_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    FLAGS_enable_mla = true;
    FLAGS_block_size = 16;
    const int32_t device_index = rank % dev_count;
    xllm::Device xllm_device(device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();
    auto process_group =
        create_test_process_group(rank, world_size, port, host, device);
    CHECK(process_group) << "Rank " << rank
                         << ": failed to create process group";

    ParallelArgs parallel_args(rank, world_size, process_group.get());
    parallel_args.tp_group_ = process_group.get();
    parallel_args.sp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);
    ModelArgs model_args = create_attention_model_args(/*q_lora_rank=*/64,
                                                       /*enable_indexer=*/true);
    QuantArgs quant_args = create_default_quant_args();
    StateDict state_dict = create_attention_state_dict(model_args, options);

    constexpr int32_t prefix_len = 8;
    constexpr int32_t chunk1_len = 4;
    constexpr int32_t chunk2_len = 4;
    torch::Tensor prefix_tokens =
        torch::arange(prefix_len, options.dtype(torch::kInt32).device(device));
    torch::Tensor prefix_hidden_states =
        seeded_tensor("attention_multi_device/chunked_prefix_hidden_states",
                      {prefix_len, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor prefix_positions =
        torch::arange(prefix_len, options.dtype(torch::kInt32).device(device));

    torch::Tensor chunk1_tokens =
        torch::arange(prefix_len,
                      prefix_len + chunk1_len,
                      options.dtype(torch::kInt32).device(device));
    torch::Tensor chunk1_hidden_states =
        seeded_tensor("attention_multi_device/chunked_suffix1_hidden_states",
                      {chunk1_len, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor chunk1_positions =
        torch::arange(prefix_len,
                      prefix_len + chunk1_len,
                      options.dtype(torch::kInt32).device(device));

    torch::Tensor chunk2_tokens =
        torch::arange(prefix_len + chunk1_len,
                      prefix_len + chunk1_len + chunk2_len,
                      options.dtype(torch::kInt32).device(device));
    torch::Tensor chunk2_hidden_states =
        seeded_tensor("attention_multi_device/chunked_suffix2_hidden_states",
                      {chunk2_len, model_args.hidden_size()},
                      torch::kBFloat16,
                      device);
    torch::Tensor chunk2_positions =
        torch::arange(prefix_len + chunk1_len,
                      prefix_len + chunk1_len + chunk2_len,
                      options.dtype(torch::kInt32).device(device));

    KVCache sharded_kv_cache = create_decode_kv_cache(model_args, options);
    sharded_kv_cache.get_k_cache().zero_();
    KVCache full_weight_kv_cache = create_decode_kv_cache(model_args, options);
    full_weight_kv_cache.get_k_cache().zero_();

    auto [sharded_prefix_output, sharded_prefix_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   prefix_tokens,
                                   prefix_positions,
                                   prefix_hidden_states,
                                   sharded_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false);
    auto [full_prefix_output, full_prefix_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   prefix_tokens,
                                   prefix_positions,
                                   prefix_hidden_states,
                                   full_weight_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false);
    check_tensors_close(sharded_prefix_output,
                        full_prefix_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention chunked prefix output");
    check_tensors_close(sharded_prefix_k_cache,
                        full_prefix_k_cache,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention chunked prefix k_cache");

    auto [sharded_output, sharded_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   chunk1_tokens,
                                   chunk1_positions,
                                   chunk1_hidden_states,
                                   sharded_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false,
                                   BatchForwardType::CHUNKED_PREFILL,
                                   prefix_len);
    auto [full_weight_output, full_weight_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   chunk1_tokens,
                                   chunk1_positions,
                                   chunk1_hidden_states,
                                   full_weight_kv_cache,
                                   true,
                                   /*enable_fused_mla_kernel=*/false,
                                   BatchForwardType::CHUNKED_PREFILL,
                                   prefix_len);

    xllm_device.synchronize_default_stream();
    check_tensors_close(sharded_output,
                        full_weight_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention chunked prefill output");
    check_tensors_close(sharded_k_cache,
                        full_weight_k_cache,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention chunked prefill k_cache");
    check_tensors_close(sharded_kv_cache.get_index_cache(),
                        full_weight_kv_cache.get_index_cache(),
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention chunked prefill index_cache");

    auto [sharded_second_output, sharded_second_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   chunk2_tokens,
                                   chunk2_positions,
                                   chunk2_hidden_states,
                                   sharded_kv_cache,
                                   false,
                                   /*enable_fused_mla_kernel=*/false,
                                   BatchForwardType::CHUNKED_PREFILL,
                                   prefix_len + chunk1_len);
    auto [full_weight_second_output, full_weight_second_k_cache] =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   chunk2_tokens,
                                   chunk2_positions,
                                   chunk2_hidden_states,
                                   full_weight_kv_cache,
                                   true,
                                   /*enable_fused_mla_kernel=*/false,
                                   BatchForwardType::CHUNKED_PREFILL,
                                   prefix_len + chunk1_len);

    xllm_device.synchronize_default_stream();
    check_tensors_close(sharded_second_output,
                        full_weight_second_output,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention second chunked prefill output");
    check_tensors_close(sharded_second_k_cache,
                        full_weight_second_k_cache,
                        /*rtol=*/1e-3,
                        /*atol=*/1e-3,
                        "DeepseekV2Attention second chunked prefill k_cache");
    check_tensors_close(
        sharded_kv_cache.get_index_cache(),
        full_weight_kv_cache.get_index_cache(),
        /*rtol=*/1e-3,
        /*atol=*/1e-3,
        "DeepseekV2Attention second chunked prefill index_cache");
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

}  // namespace

class AttentionMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    port_ = 29521;
    host_ = "127.0.0.1";
  }

  void run_test(bool enable_fused_mla_kernel, int64_t q_lora_rank = 0) {
    std::vector<pid_t> child_pids;
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        const int32_t exit_code =
            run_attention_test_child(rank,
                                     world_size_,
                                     port_,
                                     host_,
                                     enable_fused_mla_kernel,
                                     q_lora_rank);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        const int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed)
          << "Attention multi-device equivalence test failed.";
    }
  }

  void run_prefill_test(bool use_glm5_args = false) {
    std::vector<pid_t> child_pids;
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        const int32_t exit_code = run_attention_prefill_test_child(
            rank, world_size_, port_, host_, use_glm5_args);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        const int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed) << "Attention multi-device prefill test failed.";
    }
  }

  void run_prefill_fallback_test(bool use_glm5_args = false) {
    std::vector<pid_t> child_pids;
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        const int32_t exit_code = run_attention_prefill_fallback_test_child(
            rank, world_size_, port_, host_, use_glm5_args);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        const int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed)
          << "Attention multi-device prefill fallback test failed.";
    }
  }

  void run_chunked_test() {
    std::vector<pid_t> child_pids;
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        const int32_t exit_code =
            run_attention_chunked_test_child(rank, world_size_, port_, host_);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        const int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed) << "Attention multi-device chunked test failed.";
    }
  }

  int32_t world_size_;
  int32_t port_;
  std::string host_;
};

TEST_F(AttentionMultiDeviceTest, DecodeFullWeightMatchesShardedPath) {
  run_test(/*enable_fused_mla_kernel=*/false);
}

TEST_F(AttentionMultiDeviceTest, DecodeFullWeightMatchesShardedPathFusedMLA) {
  run_test(/*enable_fused_mla_kernel=*/true, /*q_lora_rank=*/64);
}

TEST_F(AttentionMultiDeviceTest, PrefillFullWeightMatchesShardedPath) {
  run_prefill_test();
}

TEST_F(AttentionMultiDeviceTest, Glm5PrefillFullWeightMatchesShardedPath) {
  run_prefill_test(/*use_glm5_args=*/true);
}

TEST_F(AttentionMultiDeviceTest, PrefillFullWeightFallbackMatchesShardedPath) {
  run_prefill_fallback_test();
}

TEST_F(AttentionMultiDeviceTest,
       Glm5PrefillFullWeightFallbackMatchesShardedPath) {
  run_prefill_fallback_test(/*use_glm5_args=*/true);
}

TEST_F(AttentionMultiDeviceTest, ChunkedPrefillFullWeightMatchesShardedPath) {
  run_chunked_test();
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
