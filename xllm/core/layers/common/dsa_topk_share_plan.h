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

inline bool should_compute_topk_from_pattern(const std::string& pattern,
                                             int32_t layer_id) {
  CHECK_GE(layer_id, 0) << "DSA top-k sharing layer id must be non-negative.";
  CHECK_LT(layer_id, static_cast<int32_t>(pattern.size()))
      << "DSA top-k sharing pattern is shorter than num_hidden_layers.";
  const char symbol =
      static_cast<char>(std::toupper(static_cast<unsigned char>(
          pattern[static_cast<size_t>(layer_id)])));
  CHECK(symbol == 'F' || symbol == 'S')
      << "DSA top-k sharing pattern only supports F/S, got " << symbol;
  return symbol == 'F';
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

  bool compute_topk = true;
  if (!args.index_topk_pattern().empty()) {
    compute_topk =
        should_compute_topk_from_pattern(args.index_topk_pattern(), layer_id);
  } else {
    const int32_t freq = args.index_topk_freq();
    CHECK_GT(freq, 1) << "DSA top-k sharing freq must be greater than 1.";
    const int32_t offset = args.index_skip_topk_offset();
    CHECK_GE(offset, 0) << "DSA top-k sharing offset must be non-negative.";
    compute_topk = layer_id <= offset || (layer_id - offset) % freq == 0;
  }

  decision.reuse_topk = !compute_topk;
  decision.output_topk = compute_topk;
  return decision;
}

}  // namespace xllm::layer
