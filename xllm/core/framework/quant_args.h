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
#include <string>

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
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", is_sym: " << args.is_sym();
  os << ", activation_dynamic: " << args.activation_dynamic();
  os << ", fmt: " << args.fmt();
  os << "]";
  return os;
}

}  // namespace xllm
