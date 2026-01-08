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

#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

// ============================================================================
// FP8 Fused Weight Utilities
// ============================================================================
// Unlike INT8/SmoothQuant (per-channel), FP8 usually uses per-tensor scaling.
// When fusing separate layers (e.g., gate_proj + up_proj) into one, we cannot
// simply concatenate them if they have different scaling factors. We must
// requantize all partitions to align with a single global max_scale.

struct Fp8PartitionInfo {
  std::vector<float> scales;
  std::vector<int64_t> logical_widths;

  bool empty() const { return scales.empty(); }
  size_t size() const { return scales.size(); }
};

inline float compute_max_scale(const std::vector<float>& scales) {
  if (scales.empty()) {
    return 1.0f;
  }
  return *std::max_element(scales.begin(), scales.end());
}

// Detect if the checkpoint contains valid separate scales for each partition.
// The check on the last element serves as a heuristic to ensure the scales
// are fully populated and not just initialized to a sentinel/minimum value.
inline bool is_unfused_checkpoint(const std::vector<float>& scales) {
  return scales.size() > 1 &&
         scales.back() > std::numeric_limits<float>::lowest();
}

// Realigns FP8 partitions to a unified global scale to enable fusion.
// Logic:
// 1. Recover original values (FP8 -> FP16) using partition-specific scales.
// 2. Re-quantize (FP16 -> FP8) using the new global max_scale.
void requantize_fp8_weight(torch::Tensor& weight,
                           const std::vector<float>& partition_scales,
                           const std::vector<int64_t>& logical_widths,
                           float max_scale) {
  if (partition_scales.size() != logical_widths.size()) {
    return;
  }

  int64_t start = 0;
  for (size_t idx = 0; idx < logical_widths.size(); ++idx) {
    int64_t logical_width = logical_widths[idx];
    if (logical_width == 0) {
      continue;
    }
    int64_t end = start + logical_width;

    // Dequantize: FP8 -> FP16 with original scale
    auto weight_slice = weight.slice(0, start, end);
    auto weight_fp16 = weight_slice.to(torch::kFloat16) * partition_scales[idx];

    // Requantize: FP16 -> FP8 with unified max_scale
    auto weight_scaled = weight_fp16 / max_scale;
    auto weight_quantized = weight_scaled.to(torch::kFloat8_e4m3fn);

    weight.slice(0, start, end).copy_(weight_quantized);
    start = end;
  }
}

// Load max input scale from multiple prefixes
torch::Tensor load_max_input_scale(const StateDict& state_dict,
                                   const std::vector<std::string>& prefixes) {
  torch::Tensor max_scale;
  for (const auto& prefix : prefixes) {
    auto scale_tensor = state_dict.get_tensor(prefix + "input_scale");
    if (scale_tensor.defined()) {
      auto scale_val = scale_tensor.flatten().max();
      if (!max_scale.defined()) {
        max_scale = scale_val;
      } else {
        max_scale = torch::max(max_scale, scale_val);
      }
    }
  }
  return max_scale;
}

// ============================================================================
// FP8 Forward Helper
// ============================================================================
// Performs FP8 W8A8 quantized linear: input quantization + scaled matmul.
// Consolidates repeated logic from Column/QKV/RowParallelLinear forward paths.

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  // Flatten input to 2D for quantization
  auto input_2d = input.view({-1, input.size(-1)});

  // Quantize input to FP8 (static or dynamic based on input_scale presence)
  xllm::kernel::Fp8ScaledQuantizeParams quantize_params;
  quantize_params.input = input_2d;
  quantize_params.output = std::nullopt;
  quantize_params.scale = input_scale;

  auto [quantized_input, a_scale] =
      xllm::kernel::fp8_scaled_quantize(quantize_params);

  // FP8 scaled matmul
  xllm::kernel::Fp8ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = a_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.bias = bias;
  matmul_params.output = std::nullopt;
  matmul_params.output_dtype = output_dtype;
  matmul_params.input_shape = input.sizes().vec();

  return xllm::kernel::fp8_scaled_matmul(matmul_params);
}

}  // namespace

ColumnParallelLinearImpl::ColumnParallelLinearImpl(const ModelContext& context)
    : ColumnParallelLinearImpl(context.get_model_args().hidden_size(),
                               context.get_model_args().vocab_size(),
                               /*bias=*/false,
                               /*gather_output=*/true,
                               QuantArgs{},
                               context.get_parallel_args().tp_group_,
                               context.get_tensor_options()) {}

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const FusedLinearExtraArgs& linear_extra_args)
    : gather_output_(gather_output),
      quant_args_(quant_args),
      process_group_(process_group),
      device_(options.device()),
      linear_extra_args_(linear_extra_args) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(out_features % world_size_ == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size_;
  const int64_t out_features_per_partition = out_features / world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({in_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale is per-tensor (scalar)
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({1}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
    // output dtype for scaled matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    // Quantize input tensor with int8 in 'dynamic_per_token' mode using
    // scaled_quantize
    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    // For SmoothQuant, use scaled_matmul with quantization parameters
    // for now, we only support w8a8 quantization
    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization
    LOG_IF(FATAL, quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else {
    // For unquantized case, use regular matmul
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);
    // for input, there is one smooth value
    LOAD_WEIGHT(smooth);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    // Find the first available "smooth" tensor in prefixes (e.g.,
    // "gate.smooth", "up_proj.smooth", etc.)
    for (const auto& prefix : prefixes) {
      auto smooth_tensor_candidate = state_dict.get_tensor(prefix + "smooth");
      if (smooth_tensor_candidate.defined()) {
        // Copy the found smooth tensor to the module parameter
        CHECK_EQ(smooth_.sizes(), smooth_tensor_candidate.sizes())
            << "smooth weight size mismatch for " << state_dict.prefix()
            << "smooth";
        smooth_.copy_(smooth_tensor_candidate);
        smooth_is_loaded_ = true;
        break;
      }
    }
    LOAD_FUSED_WEIGHT(qweight, 0);
    LOAD_FUSED_WEIGHT(per_channel_scale, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 fused layer loading: each partition may have its own per-tensor scale
    // (unfused checkpoint). We must requantize all partitions with max_scale.

    // Step 1: Collect partition info BEFORE LOAD_FUSED_WEIGHT (clears list)
    Fp8PartitionInfo partition_info;
    if (!weight_scale_is_loaded_) {
      for (const auto& prefix : prefixes) {
        auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
        if (scale_tensor.defined()) {
          partition_info.scales.push_back(scale_tensor.flatten().item<float>());
        }
        auto weight_tensor = state_dict.get_sharded_tensor(
            prefix + "weight", 0, rank, world_size);
        if (weight_tensor.defined()) {
          partition_info.logical_widths.push_back(weight_tensor.size(0));
        }
      }
    }

    // Step 2: Load fused weight
    LOAD_FUSED_WEIGHT(weight, 0);

    // Step 3: Requantize if needed (unfused checkpoint case)
    if (!weight_scale_is_loaded_ && !partition_info.empty()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales) && weight_.defined() &&
          partition_info.logical_widths.size() ==
              partition_info.scales.size()) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }

      weight_scale_.fill_(max_scale);
      weight_scale_is_loaded_ = true;
    }

    // Step 4: Load input_scale for static activation quantization
    if (!quant_args_.activation_dynamic() && input_scale_.defined() &&
        !input_scale_is_loaded_) {
      auto max_input_scale = load_max_input_scale(state_dict, prefixes);
      if (max_input_scale.defined()) {
        input_scale_.copy_(max_input_scale.view({1}));
        input_scale_is_loaded_ = true;
      }
    }
  } else {
    LOAD_FUSED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

QKVParallelLinearImpl::QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale: create {3} for Q/K/V, will use max() after loading
    // load separate scales then merge with max
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({3}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    // Also create {3} for Q/K/V, will use max() after loading
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({3}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
    // output dtype for scaled matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization
    LOG_IF(FATAL, quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    // Use max of Q/K/V scales as unified scale for fused projection
    // Note: weight_scale_ and input_scale_ are already scalar tensors
    // (replaced with max values in load_state_dict)
    auto a_scale = input_scale_.defined()
                       ? std::optional<torch::Tensor>(input_scale_)
                       : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, a_scale, bias, output_dtype_);
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;

    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

void QKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  LOAD_QKV_WEIGHT(weight, 0, num_kv_head_replicas_);
  if (bias_.defined()) {
    LOAD_QKV_WEIGHT(bias, 0, num_kv_head_replicas_);
  }
  // FP8: load weight_scale and input_scale, requantize if needed
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    // Build partition info for Q/K/V
    Fp8PartitionInfo partition_info;
    int64_t num_heads_per_partition = num_heads_ / world_size_;
    int64_t num_kv_heads_per_partition = num_kv_heads_ / world_size_;
    partition_info.logical_widths = {
        num_heads_per_partition * head_size_,      // Q
        num_kv_heads_per_partition * head_size_,   // K
        num_kv_heads_per_partition * head_size_};  // V

    for (const auto& prefix : prefixes) {
      auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
      if (scale_tensor.defined()) {
        partition_info.scales.push_back(scale_tensor.flatten().item<float>());
      }
    }

    // Requantize if unfused checkpoint (multiple scales)
    if (partition_info.scales.size() > 1 && weight_.defined()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales)) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }
      weight_scale_.fill_(max_scale);
    } else if (partition_info.scales.size() == 1) {
      weight_scale_.fill_(partition_info.scales[0]);
    } else {
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    }

    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_FUSED_WEIGHT(input_scale, 0);
    }

    // For per-tensor quantization with fused QKV, replace scale tensors with
    // scalar max values to avoid recomputing max() in every forward() call.
    // Only apply for per-tensor quantization.
    // Per-channel/per-block quantization should NOT take max.
    if (weight_scale_.defined() && weight_scale_.numel() > 1) {
      weight_scale_ = weight_scale_.max();
    }
    if (input_scale_.defined() && input_scale_.numel() > 1) {
      input_scale_ = input_scale_.max();
    }
  }
}

void QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int32_t shard_tensor_count = 3;
  const int64_t shard_size = num_heads_ * head_size_;
  CHECK_EQ(num_heads_, num_kv_heads_);
  LOAD_MERGED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const FusedLinearExtraArgs& linear_extra_args)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      process_group_(process_group),
      linear_extra_args_(linear_extra_args) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ = register_parameter(
        "per_channel_scale",
        torch::empty({out_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    smooth_ = register_parameter("smooth",
                                 torch::empty({in_features_per_partition},
                                              options.dtype(torch::kFloat32)),
                                 /*requires_grad=*/false);
    // Output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale is per-tensor (scalar)
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({1}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
    // output dtype for scaled matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  if (!input_is_parallelized_) {
    input = xllm::parallel_state::scatter(input, process_group_);
  }

  auto bias = (bias_.defined() && rank_ == 0)
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;
  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    // Call scaled_quantize: quantizes input tensor with int8 in
    // 'dynamic_per_token' mode
    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    // for now, we only support w8a8 quantization
    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization
    LOG_IF(FATAL, quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }
  if (enable_result_reduction_ && world_size_ > 1) {
    output = xllm::parallel_state::reduce(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_WEIGHT(per_channel_scale);
    LOAD_SHARDED_WEIGHT(smooth, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options) {
  weight_ =
      register_parameter("weight",
                         torch::empty({out_features, in_features}, options),
                         /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;

  auto output = xllm::kernel::matmul(matmul_params);
  return output;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
