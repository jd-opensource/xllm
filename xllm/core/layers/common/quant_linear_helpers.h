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
#include <cctype>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "kernels/ops_api.h"

namespace xllm::layer {

// ── Utility ────────────────────────────────────────────────────────────────

inline std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

// ── Quant Method Resolution ─────────────────────────────────────────────────

inline void resolve_weight_quant_method_for_linear_load(
    const QuantArgs& quant_args,
    const StateDict& state_dict,
    const std::vector<std::string>* local_prefixes,
    std::optional<std::string>& resolved_weight_quant_method) {
  const auto prefixes =
      local_prefixes == nullptr ? std::vector<std::string>{} : *local_prefixes;
  auto resolved =
      quant_args.get_quant_method_from_prefixes(state_dict, prefixes);
  if (resolved.has_value()) {
    resolved_weight_quant_method = to_lower_copy(resolved.value());
    return;
  }
  if (!quant_args.quant_descs().empty()) {
    LOG(WARNING) << "[LinearLoad][QuantMethod] quant_descs is not empty but "
                    "quant method was not resolved from state_dict prefixes. "
                    "state_dict.prefix="
                 << state_dict.prefix();
  }
  resolved_weight_quant_method = std::nullopt;
}

// ── Quant Method Checks ─────────────────────────────────────────────────────

inline bool is_w8a8_dynamic_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8_dynamic";
}

inline bool is_w8a8_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8";
}

// ── W8A8 Dequant Scale Dtype ───────────────────────────────────────────────

inline torch::Dtype get_w8a8_deq_scale_dtype(
    const torch::TensorOptions& options) {
  const torch::Dtype dtype = c10::typeMetaToScalarType(options.dtype());
  if (dtype == torch::kFloat16) {
    return torch::kInt64;
  }
  if (dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  LOG(WARNING) << "W8A8 deq_scale defaults to float32 for dtype " << dtype;
  return torch::kFloat32;
}

// ── W8A8 Linear Parameter Refs ─────────────────────────────────────────────

struct W8A8LinearParamRefs {
  torch::Tensor& weight;
  bool& weight_is_loaded;
  torch::Tensor& input_scale;
  bool& input_scale_is_loaded;
  torch::Tensor& input_offset;
  bool& input_offset_is_loaded;
  torch::Tensor& deq_scale;
  bool& deq_scale_is_loaded;
  torch::Tensor& quant_bias;
  bool& quant_bias_is_loaded;
  torch::Tensor& weight_scale;
  bool& weight_scale_is_loaded;
  torch::Tensor& weight_offset;
  bool& weight_offset_is_loaded;
};

// ── Lazy Parameter Registration ────────────────────────────────────────────

inline void ensure_w8a8_params_for_linear_load(
    torch::nn::Module* module,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const std::optional<std::string>& resolved_weight_quant_method,
    int64_t shared_input_param_size,
    W8A8LinearParamRefs refs) {
  std::vector<weight::LazyParameterSpec> specs;
  auto push = [&](torch::Tensor& tensor,
                  bool& tensor_is_loaded,
                  const char* name,
                  std::vector<int64_t> sizes,
                  const torch::TensorOptions& tensor_options) {
    specs.push_back(weight::LazyParameterSpec{
        &tensor, &tensor_is_loaded, name, std::move(sizes), tensor_options});
  };

  if (!is_w8a8_quant(resolved_weight_quant_method) &&
      !is_w8a8_dynamic_quant(resolved_weight_quant_method)) {
    if (!quant_args.quant_descs().empty()) {
      CHECK(refs.weight.defined())
          << "weight must be registered before lazy quant fallback";
      const int64_t out_features = refs.weight.size(0);
      const int64_t in_features = refs.weight.size(1);
      specs.reserve(1);
      push(refs.weight,
           refs.weight_is_loaded,
           "weight",
           {out_features, in_features},
           options);
      weight::ensure_parameter_storage(module, specs);
    }
    return;
  }

  CHECK(refs.weight.defined())
      << "weight must be registered before lazy quant init";
  const int64_t out_features = refs.weight.size(0);
  const int64_t in_features = refs.weight.size(1);

  specs.reserve(4);
  if (is_w8a8_quant(resolved_weight_quant_method)) {
    push(refs.input_scale,
         refs.input_scale_is_loaded,
         "input_scale",
         {shared_input_param_size},
         options.dtype(torch::kFloat32));
    push(refs.input_offset,
         refs.input_offset_is_loaded,
         "input_offset",
         {shared_input_param_size},
         options.dtype(torch::kInt8));
    push(refs.deq_scale,
         refs.deq_scale_is_loaded,
         "deq_scale",
         {out_features},
         options.dtype(get_w8a8_deq_scale_dtype(options)));
    push(refs.quant_bias,
         refs.quant_bias_is_loaded,
         "quant_bias",
         {out_features},
         options.dtype(torch::kInt32));
  } else {
    push(refs.weight_scale,
         refs.weight_scale_is_loaded,
         "weight_scale",
         {out_features},
         options.dtype(torch::kFloat32));
    push(refs.weight_offset,
         refs.weight_offset_is_loaded,
         "weight_offset",
         {out_features},
         options.dtype(torch::kFloat32));
  }
  weight::ensure_parameter_storage(module, specs);
}

// ── NPU W8A8 Forward (Static) ──────────────────────────────────────────────

inline torch::Tensor npu_w8a8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& input_scale,
    const torch::Tensor& input_offset,
    const torch::Tensor& deq_scale,
    const std::optional<torch::Tensor>& quant_bias,
    at::ScalarType output_dtype) {
  kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  quant_params.scale = input_scale;
  quant_params.zero_point = input_offset;
  quant_params.axis = -1;

  auto quantized_input = kernel::quantize(quant_params);

  kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = deq_scale;
  quant_matmul_params.bias = quant_bias;
  quant_matmul_params.output_dtype = output_dtype;

  return kernel::quant_matmul(quant_matmul_params);
}

// ── NPU W8A8 Forward (Dynamic) ─────────────────────────────────────────────

inline torch::Tensor npu_w8a8_dynamic_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype,
    const std::optional<torch::Tensor>& weight_offset = std::nullopt) {
  kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;

  torch::Tensor quantized_input;
  std::optional<torch::Tensor> pertoken_scale;
  std::tie(quantized_input, pertoken_scale) =
      kernel::dynamic_quant(quant_params);
  CHECK(pertoken_scale.has_value() && pertoken_scale->defined())
      << "dynamic_quant must return per-token scale for w8a8_dynamic.";

  kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = weight_scale;
  quant_matmul_params.pertoken_scale = pertoken_scale->reshape({-1});
  quant_matmul_params.output_dtype = output_dtype;
  if (weight_offset.has_value() && weight_offset->defined()) {
    quant_matmul_params.offset = weight_offset;
  }
  if (bias.has_value() && bias->defined()) {
    quant_matmul_params.bias = bias;
  }
  return kernel::quant_matmul(quant_matmul_params);
}

}  // namespace xllm::layer
