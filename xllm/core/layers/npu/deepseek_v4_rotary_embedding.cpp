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

#include "deepseek_v4_rotary_embedding.h"

#include <glog/logging.h>
#include <torch/nn/functional/embedding.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <utility>

#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

namespace {

constexpr const char* kDefaultGroup = "default";
constexpr const char* kC4Group = "c4";
constexpr const char* kC128Group = "c128";

std::string normalize_group_name(const std::string& group_name) {
  std::string normalized = group_name;
  boost::to_lower(normalized);
  if (normalized == "c4" || normalized == "c_4") {
    return kC4Group;
  }
  if (normalized == "c128" || normalized == "c_128") {
    return kC128Group;
  }
  return normalized;
}

}  // namespace

DeepseekV4RotaryEmbedding::DeepseekV4RotaryEmbedding(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    bool interleaved,
    float rope_theta,
    float compress_rope_theta,
    float scaling_factor,
    float extrapolation_factor,
    int64_t beta_fast,
    int64_t beta_slow,
    float attn_factor,
    float mscale,
    float mscale_all_dim,
    int64_t original_max_position_embeddings,
    const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim),
      max_position_embeddings_(max_position_embeddings),
      interleaved_(interleaved),
      options_(options) {
  CHECK_GT(rotary_dim_, 0) << "rotary_dim must be > 0";
  CHECK_EQ(rotary_dim_ % 2, 0) << "rotary_dim must be even";
  CHECK_GT(max_position_embeddings_, 0)
      << "max_position_embeddings must be > 0";

  const int64_t max_idx_default = max_position_embeddings_;
  const int64_t max_idx_c4 = (max_position_embeddings_ - 1) / 4 + 1;
  const int64_t max_idx_c128 = (max_position_embeddings_ - 1) / 128 + 1;

  cos_sin_cache_by_group_[kDefaultGroup] =
      create_cos_sin_cache(max_idx_default,
                           rope_theta,
                           scaling_factor,
                           extrapolation_factor,
                           beta_fast,
                           beta_slow,
                           attn_factor,
                           mscale,
                           mscale_all_dim,
                           original_max_position_embeddings);

  cos_sin_cache_by_group_[kC4Group] =
      create_cos_sin_cache(max_idx_c4,
                           compress_rope_theta,
                           scaling_factor,
                           extrapolation_factor,
                           beta_fast,
                           beta_slow,
                           attn_factor,
                           mscale,
                           mscale_all_dim,
                           original_max_position_embeddings);
  cos_sin_cache_by_group_[kC128Group] =
      create_cos_sin_cache(max_idx_c128,
                           compress_rope_theta,
                           scaling_factor,
                           extrapolation_factor,
                           beta_fast,
                           beta_slow,
                           attn_factor,
                           mscale,
                           mscale_all_dim,
                           original_max_position_embeddings);
}

DeepseekV4RotaryEmbedding::GroupCosSinMap DeepseekV4RotaryEmbedding::build(
    const std::unordered_map<std::string, torch::Tensor>& positions_map) const {
  GroupCosSinMap result;
  namespace F = torch::nn::functional;

  for (const auto& it : positions_map) {
    std::string group_name = normalize_group_name(it.first);
    auto cache_it = cos_sin_cache_by_group_.find(group_name);
    if (cache_it == cos_sin_cache_by_group_.end()) {
      continue;
    }
    const auto& positions = it.second;
    auto cos_sin = F::embedding(positions, cache_it->second);
    auto chunks = cos_sin.chunk(2, -1);
    result[group_name] =
        std::make_pair(chunks[0].contiguous(), chunks[1].contiguous());
  }

  return result;
}

DeepseekV4RotaryEmbedding::GroupCosSinMap DeepseekV4RotaryEmbedding::build(
    const torch::Tensor& default_positions) const {
  return build({{kDefaultGroup, default_positions}});
}

void DeepseekV4RotaryEmbedding::register_layer(
    const std::string& layer_name,
    const std::vector<std::string>& groups) {
  std::vector<std::string> normalized_groups;
  normalized_groups.reserve(groups.size());

  for (const auto& group : groups) {
    auto normalized = normalize_group_name(group);
    if (cos_sin_cache_by_group_.find(normalized) ==
        cos_sin_cache_by_group_.end()) {
      continue;
    }
    normalized_groups.push_back(normalized);
  }

  if (normalized_groups.empty()) {
    normalized_groups.push_back(kDefaultGroup);
  }
  layer_groups_[layer_name] = std::move(normalized_groups);
}

DeepseekV4RotaryEmbedding::GroupCosSinMap
DeepseekV4RotaryEmbedding::select_layer_groups(
    const std::string& layer_name,
    const GroupCosSinMap& group_cos_sin) const {
  GroupCosSinMap selected;

  auto layer_it = layer_groups_.find(layer_name);
  if (layer_it == layer_groups_.end()) {
    auto default_it = group_cos_sin.find(kDefaultGroup);
    if (default_it != group_cos_sin.end()) {
      selected[kDefaultGroup] = default_it->second;
    }
    return selected;
  }

  for (const auto& group_name : layer_it->second) {
    auto group_it = group_cos_sin.find(group_name);
    if (group_it != group_cos_sin.end()) {
      selected[group_name] = group_it->second;
    }
  }
  return selected;
}

torch::Tensor DeepseekV4RotaryEmbedding::get_cos_sin_cache(
    const std::string& group_name) const {
  auto normalized = normalize_group_name(group_name);
  auto it = cos_sin_cache_by_group_.find(normalized);
  CHECK(it != cos_sin_cache_by_group_.end())
      << "unsupported deepseek v4 rope group: " << group_name;
  return it->second;
}

std::vector<std::string> DeepseekV4RotaryEmbedding::registered_groups() const {
  std::vector<std::string> groups;
  groups.reserve(cos_sin_cache_by_group_.size());
  for (const auto& it : cos_sin_cache_by_group_) {
    groups.push_back(it.first);
  }
  std::sort(groups.begin(), groups.end());
  return groups;
}

torch::Tensor DeepseekV4RotaryEmbedding::create_cos_sin_cache(
    int64_t max_position_embeddings,
    float theta,
    float scaling_factor,
    float extrapolation_factor,
    int64_t beta_fast,
    int64_t beta_slow,
    float attn_factor,
    float mscale,
    float mscale_all_dim,
    int64_t original_max_position_embeddings) const {
  auto inv_freq = rotary::apply_deepseek_yarn_rope_scaling(
      scaling_factor,
      extrapolation_factor,
      beta_fast,
      beta_slow,
      rotary_dim_,
      theta,
      original_max_position_embeddings);

  return rotary::compute_cos_sin_cache(rotary_dim_,
                                       max_position_embeddings,
                                       interleaved_,
                                       scaling_factor,
                                       attn_factor,
                                       mscale,
                                       mscale_all_dim,
                                       inv_freq,
                                       options_);
}

}  // namespace layer
}  // namespace xllm
