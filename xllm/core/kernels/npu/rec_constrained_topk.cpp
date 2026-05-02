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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr double kInvalidLogit = -1.0e20;

torch::Tensor index_table_by_positions(const torch::Tensor& values,
                                       const torch::Tensor& positions) {
  if (values.size(0) == 0) {
    return torch::zeros(positions.sizes(), values.options());
  }
  torch::Tensor flat_positions = positions.reshape({-1}).to(torch::kLong);
  return values.index_select(/*dim=*/0, flat_positions).view(positions.sizes());
}

std::tuple<torch::Tensor, torch::Tensor> build_range_candidates(
    const torch::Tensor& values,
    const torch::Tensor& begins,
    const torch::Tensor& degrees,
    int64_t candidate_width) {
  CHECK_GT(candidate_width, 0);
  const int64_t num_rows = begins.size(0);
  auto index_options =
      torch::TensorOptions().dtype(torch::kLong).device(begins.device());
  torch::Tensor offsets =
      torch::arange(candidate_width, index_options).unsqueeze(/*dim=*/0);
  torch::Tensor positions = begins.to(torch::kLong).unsqueeze(/*dim=*/1) +
                            offsets.expand({num_rows, candidate_width});
  torch::Tensor valid =
      offsets.lt(degrees.to(torch::kLong).unsqueeze(/*dim=*/1));
  const int64_t max_position = std::max<int64_t>(values.size(0) - 1, 0);
  torch::Tensor safe_positions =
      torch::where(valid, positions, torch::zeros_like(positions))
          .clamp(/*min=*/0, /*max=*/max_position);
  torch::Tensor candidate_tokens =
      index_table_by_positions(values, safe_positions);
  return {candidate_tokens, valid};
}

std::tuple<torch::Tensor, torch::Tensor> finish_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& candidate_tokens,
    const torch::Tensor& valid_candidates,
    const torch::Tensor& temperatures,
    int64_t top_k) {
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D, got " << logits.sizes();
  CHECK_EQ(candidate_tokens.dim(), 2)
      << "candidate_tokens must be 2-D, got " << candidate_tokens.sizes();
  CHECK_EQ(candidate_tokens.sizes(), valid_candidates.sizes())
      << "candidate token/mask shape mismatch, tokens="
      << candidate_tokens.sizes() << ", valid=" << valid_candidates.sizes();
  CHECK_EQ(logits.size(0), candidate_tokens.size(0))
      << "logits/candidate row mismatch, logits=" << logits.sizes()
      << ", candidates=" << candidate_tokens.sizes();
  CHECK_LE(top_k, candidate_tokens.size(1))
      << "top_k exceeds candidate width, top_k=" << top_k
      << ", candidate_width=" << candidate_tokens.size(1);

  torch::Tensor scaled_logits = logits.to(torch::kFloat32);
  if (temperatures.defined()) {
    torch::Tensor temps =
        temperatures.to(torch::kFloat32).to(logits.device()).unsqueeze(1);
    temps = torch::where(temps == 0, torch::ones_like(temps), temps);
    scaled_logits = scaled_logits / temps;
  }

  torch::Tensor token_indices = candidate_tokens.to(torch::kLong);
  torch::Tensor candidate_logits =
      scaled_logits.gather(/*dim=*/1, /*index=*/token_indices);
  candidate_logits = candidate_logits.masked_fill(
      valid_candidates.logical_not(), kInvalidLogit);

  torch::Tensor top_values;
  torch::Tensor top_indices;
  std::tie(top_values, top_indices) = candidate_logits.topk(
      top_k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  const std::vector<int64_t> reduce_dims{1};
  torch::Tensor log_denom =
      torch::logsumexp(candidate_logits, reduce_dims, /*keepdim=*/true);
  torch::Tensor top_logprobs = top_values - log_denom;
  top_logprobs = torch::where(top_values.le(kInvalidLogit / 2.0),
                              torch::full_like(top_logprobs, kInvalidLogit),
                              top_logprobs);
  torch::Tensor top_tokens =
      candidate_tokens.gather(/*dim=*/1, /*index=*/top_indices)
          .to(torch::kLong);
  return {top_tokens, top_logprobs};
}

std::tuple<torch::Tensor, torch::Tensor> build_first_token_candidates(
    const torch::Tensor& logits,
    const torch::Tensor& first_token_ids,
    int64_t top_k) {
  const int64_t allowed_count = first_token_ids.size(0);
  const int64_t candidate_width = std::max<int64_t>(allowed_count, top_k);
  auto begin_options =
      torch::TensorOptions().dtype(torch::kLong).device(logits.device());
  torch::Tensor begins = torch::zeros({logits.size(0)}, begin_options);
  torch::Tensor degrees =
      torch::full({logits.size(0)}, allowed_count, begin_options);
  return build_range_candidates(
      first_token_ids, begins, degrees, candidate_width);
}

std::tuple<torch::Tensor, torch::Tensor> build_prefix1_candidates(
    const torch::Tensor& sequence_group,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    int64_t top_k,
    int64_t max_prefix1_degree) {
  const int64_t candidate_width = std::max<int64_t>(top_k, max_prefix1_degree);
  torch::Tensor sequence_group_flat =
      sequence_group.reshape({-1, sequence_group.size(-1)});
  torch::Tensor t0 =
      sequence_group_flat.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor begins = prefix1_offsets.index_select(/*dim=*/0, t0);
  torch::Tensor ends = prefix1_offsets.index_select(/*dim=*/0, t0 + 1);
  return build_range_candidates(
      prefix1_values, begins, ends - begins, candidate_width);
}

std::tuple<torch::Tensor, torch::Tensor> build_prefix2_candidates(
    const torch::Tensor& sequence_group,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    int64_t top_k,
    int64_t max_prefix1_degree,
    int64_t max_prefix2_degree) {
  const int64_t prefix1_width = std::max<int64_t>(top_k, max_prefix1_degree);
  torch::Tensor sequence_group_flat =
      sequence_group.reshape({-1, sequence_group.size(-1)});
  const int64_t num_rows = sequence_group_flat.size(0);
  if (prefix2_value_offsets.size(0) < 2 || prefix2_values.size(0) == 0) {
    const int64_t prefix2_width = std::max<int64_t>(top_k, max_prefix2_degree);
    torch::Tensor candidate_tokens =
        torch::zeros({num_rows, prefix2_width}, prefix2_values.options());
    torch::Tensor valid_candidates =
        torch::zeros({num_rows, prefix2_width},
                     torch::TensorOptions()
                         .dtype(torch::kBool)
                         .device(sequence_group.device()));
    return {candidate_tokens, valid_candidates};
  }
  torch::Tensor t0 =
      sequence_group_flat.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor t1 = sequence_group_flat.select(/*dim=*/1, /*index=*/1);

  torch::Tensor prefix1_begins = prefix1_offsets.index_select(/*dim=*/0, t0);
  torch::Tensor prefix1_ends = prefix1_offsets.index_select(/*dim=*/0, t0 + 1);
  torch::Tensor prefix1_candidates;
  torch::Tensor prefix1_valid;
  std::tie(prefix1_candidates, prefix1_valid) =
      build_range_candidates(prefix1_values,
                             prefix1_begins,
                             prefix1_ends - prefix1_begins,
                             prefix1_width);

  torch::Tensor matches =
      prefix1_valid.logical_and(prefix1_candidates.eq(t1.unsqueeze(1)));
  torch::Tensor relative_pair_indices =
      matches.to(torch::kLong).argmax(/*dim=*/1);
  torch::Tensor pair_valid = matches.any(/*dim=*/1);
  torch::Tensor pair_indices =
      prefix1_begins.to(torch::kLong) + relative_pair_indices;

  const int64_t max_pair_index =
      std::max<int64_t>(prefix2_value_offsets.size(0) - 2, 0);
  pair_indices =
      torch::where(pair_valid, pair_indices, torch::zeros_like(pair_indices))
          .clamp(/*min=*/0, /*max=*/max_pair_index);
  torch::Tensor prefix2_begins =
      prefix2_value_offsets.index_select(/*dim=*/0, pair_indices);
  torch::Tensor prefix2_ends =
      prefix2_value_offsets.index_select(/*dim=*/0, pair_indices + 1);
  torch::Tensor prefix2_degrees =
      torch::where(pair_valid,
                   prefix2_ends - prefix2_begins,
                   torch::zeros_like(prefix2_begins));

  const int64_t prefix2_width = std::max<int64_t>(top_k, max_prefix2_degree);
  return build_range_candidates(
      prefix2_values, prefix2_begins, prefix2_degrees, prefix2_width);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rec_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& first_token_ids,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix1_degree,
    int64_t max_prefix2_degree) {
  CHECK(logits.defined()) << "logits is required";
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D, got " << logits.sizes();
  CHECK_GT(top_k, 0);
  CHECK(first_token_ids.defined()) << "first_token_ids is required";
  CHECK(prefix1_offsets.defined()) << "prefix1_offsets is required";
  CHECK(prefix1_values.defined()) << "prefix1_values is required";
  CHECK(prefix2_value_offsets.defined()) << "prefix2_value_offsets is required";
  CHECK(prefix2_values.defined()) << "prefix2_values is required";

  torch::Tensor candidate_tokens;
  torch::Tensor valid_candidates;
  if (current_step == 0) {
    std::tie(candidate_tokens, valid_candidates) =
        build_first_token_candidates(logits, first_token_ids, top_k);
  } else if (current_step == 1) {
    CHECK(sequence_group.defined()) << "sequence_group is required for step 1";
    std::tie(candidate_tokens, valid_candidates) =
        build_prefix1_candidates(sequence_group,
                                 prefix1_offsets,
                                 prefix1_values,
                                 top_k,
                                 max_prefix1_degree);
  } else if (current_step == 2) {
    CHECK(sequence_group.defined()) << "sequence_group is required for step 2";
    std::tie(candidate_tokens, valid_candidates) =
        build_prefix2_candidates(sequence_group,
                                 prefix1_offsets,
                                 prefix1_values,
                                 prefix2_value_offsets,
                                 prefix2_values,
                                 top_k,
                                 max_prefix1_degree,
                                 max_prefix2_degree);
  } else {
    LOG(FATAL) << "Unsupported OneRec constrained step: " << current_step;
  }

  return finish_constrained_topk(
      logits, candidate_tokens, valid_candidates, temperatures, top_k);
}

}  // namespace xllm::kernel::npu
