/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <ostream>
#include <regex>
#include <string>
#include <vector>

#include "common/macros.h"

namespace xllm {

// Quantization method identifiers
static const std::string kQuantMethodFp8 = "fp8";
static const std::string kQuantMethodSmoothquant = "smoothquant";

struct QuantArgs {
  PROPERTY(std::string, quant_method);

  PROPERTY(std::string, quantize_type);
  PROPERTY(std::string, torch_dtype) = "bfloat16";
  // quantization bits
  PROPERTY(int64_t, bits) = 0;
  // MoE routed experts weight bits for DeepSeek-style SmoothQuant mixed W4A8.
  PROPERTY(int64_t, moe_weight_bits) = 8;

  // quantization group size
  PROPERTY(int64_t, group_size) = 0;

  // aka act_order, true results in better quantisation accuracy.
  PROPERTY(bool, desc_act) = false;

  // whether the input is symmetric
  PROPERTY(bool, is_sym) = false;

  // whether activation scheme is dynamic
  PROPERTY(bool, activation_dynamic) = true;

  // FP8 format : e4m3, e5m2
  PROPERTY(std::string, fmt) = "e4m3";

  // weight block size
  PROPERTY(std::vector<int64_t>, weight_block_size) = {};

  // exact module names or regexes prefixed with "re:" that should bypass
  // quantization for compressed-tensors models.
  PROPERTY(std::vector<std::string>, ignored_modules) = {};

  bool should_ignore_module(const std::string& module_name) const {
    for (const auto& pattern : ignored_modules()) {
      if (pattern == module_name) {
        return true;
      }
      if (pattern.size() > 3 && pattern.rfind("re:", 0) == 0) {
        try {
          if (std::regex_match(module_name, std::regex(pattern.substr(3)))) {
            return true;
          }
        } catch (const std::regex_error&) {
        }
      }
    }
    return false;
  }

  QuantArgs for_module(const std::string& module_name) const {
    QuantArgs local_args = *this;
    if (should_ignore_module(module_name)) {
      local_args.quant_method().clear();
    }
    return local_args;
  }

  // check if weights can be fused
  bool can_be_fused() const {
    // can't fuse quantized weights if desc_act is true
    return quant_method().empty() || !desc_act();
  }
};

inline std::ostream& operator<<(std::ostream& os, const QuantArgs& args) {
  os << "QuantArgs: [";
  os << "quant_method: " << args.quant_method();
  os << ", bits: " << args.bits();
  os << ", moe_weight_bits: " << args.moe_weight_bits();
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", is_sym: " << args.is_sym();
  os << ", activation_dynamic: " << args.activation_dynamic();
  os << ", fmt: " << args.fmt();
  os << ", ignored_modules: " << args.ignored_modules().size();
  os << "]";
  return os;
}

}  // namespace xllm
