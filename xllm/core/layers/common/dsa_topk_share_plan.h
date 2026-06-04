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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

#include <glog/logging.h>

#include "core/framework/model/model_args.h"

namespace xllm::layer {

struct DsaTopkShareDecision {
  bool reuse_topk = false;
  bool output_topk = false;
};

inline bool has_dsa_indexer(const ModelArgs& args) {
  return args.index_n_heads() > 0 && args.index_head_dim() > 0 &&
         args.index_topk() > 0;
}

inline bool should_skip_topk_from_pattern(const std::string& pattern,
                                          int32_t layer_id) {
  CHECK_GE(layer_id, 0) << "DSA top-k sharing layer id must be non-negative.";
  CHECK_LT(layer_id, static_cast<int32_t>(pattern.size()))
      << "DSA top-k sharing pattern is shorter than num_hidden_layers.";
  const char symbol =
      static_cast<char>(std::toupper(static_cast<unsigned char>(
          pattern[static_cast<size_t>(layer_id)])));
  CHECK(symbol == 'F' || symbol == 'S')
      << "DSA top-k sharing pattern only supports F/S, got " << symbol;
  return symbol == 'S';
}

inline bool should_skip_topk_from_freq(int32_t freq,
                                       int32_t offset,
                                       int32_t layer_id) {
  CHECK_GT(freq, 1) << "DSA top-k sharing freq must be greater than 1.";
  CHECK_GE(offset, 0) << "DSA top-k sharing offset must be non-negative.";
  if (offset > 0) {
    return std::max(layer_id - offset + 1, 0) % freq != 0;
  }
  return std::max(layer_id - 1, 0) % freq != 0;
}

inline bool should_next_layer_skip_topk_from_freq(int32_t freq,
                                                  int32_t offset,
                                                  int32_t layer_id) {
  CHECK_GT(freq, 1) << "DSA top-k sharing freq must be greater than 1.";
  CHECK_GE(offset, 0) << "DSA top-k sharing offset must be non-negative.";
  if (offset > 0) {
    return std::max(layer_id - offset + 2, 0) % freq != 0;
  }
  return layer_id % freq != 0;
}

inline DsaTopkShareDecision get_dsa_topk_share_decision(
    const ModelArgs& args,
    int32_t layer_id) {
  DsaTopkShareDecision decision;
  if (!has_dsa_indexer(args)) {
    return decision;
  }
  if (args.index_topk_pattern().empty() && args.index_topk_freq() <= 1) {
    return decision;
  }

  bool skip_topk = false;
  bool next_skip_topk = false;
  if (!args.index_topk_pattern().empty()) {
    const std::string& pattern = args.index_topk_pattern();
    skip_topk = should_skip_topk_from_pattern(pattern, layer_id);
    if (layer_id + 1 < static_cast<int32_t>(pattern.size())) {
      next_skip_topk = should_skip_topk_from_pattern(pattern, layer_id + 1);
    }
  } else {
    const int32_t freq = args.index_topk_freq();
    const int32_t offset = args.index_skip_topk_offset();
    skip_topk = should_skip_topk_from_freq(freq, offset, layer_id);
    next_skip_topk =
        should_next_layer_skip_topk_from_freq(freq, offset, layer_id);
  }

  decision.reuse_topk = skip_topk;
  decision.output_topk = !skip_topk && next_skip_topk;
  return decision;
}

}  // namespace xllm::layer
