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

#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"

#include <glog/logging.h>

#include <limits>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "layers/common/dp_utils.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor reshape_topk(const torch::Tensor& topk,
                           int64_t hidden_rows,
                           const char* name) {
  CHECK_GT(topk.dim(), 0) << name << " must have at least 1 dimension";
  CHECK_GT(topk.size(-1), 0) << name << " topk dimension must be positive";
  const int64_t topk_size = topk.size(-1);
  const int64_t row_count = topk.numel() / topk_size;
  CHECK_EQ(row_count, hidden_rows) << name << " row count mismatch: expected "
                                   << hidden_rows << ", actual " << row_count;
  return topk.reshape({hidden_rows, topk_size}).contiguous();
}

}  // namespace

DeepseekV4SparseMoEBlockImpl::DeepseekV4SparseMoEBlockImpl(
    const ModelArgs& model_args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    bool use_hash)
    : parallel_args_(parallel_args) {
  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = false, .use_hash = use_hash};
  moe_ = register_module(
      "moe",
      FusedMoE(model_args, moe_args, quant_args, parallel_args, options));
}

void DeepseekV4SparseMoEBlockImpl::load_state_dict(
    const StateDict& state_dict) {
  moe_->load_state_dict(state_dict);
}

void DeepseekV4SparseMoEBlockImpl::verify_loaded_weights() const {
  moe_->verify_loaded_weights();
}

bool DeepseekV4SparseMoEBlockImpl::need_gather() const {
  return need_selected_moe_dp_gather(parallel_args_);
}

ProcessGroup* DeepseekV4SparseMoEBlockImpl::routed_pg() const {
  return parallel_args_.ep_size() > 1 ? parallel_args_.moe_ep_group_
                                      : parallel_args_.tp_group_;
}

std::vector<int32_t> DeepseekV4SparseMoEBlockImpl::get_row_dp_tokens(
    int64_t hidden_rows,
    const ModelInputParams& input_params) const {
  const std::vector<int32_t>& token_nums =
      input_params.parallel.dp_global_token_nums;
  CHECK(!token_nums.empty()) << "dp_global_token_nums is empty";
  CHECK(parallel_args_.dp_local_process_group_ != nullptr)
      << "dp_local_process_group_ is not initialized";

  const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
  CHECK_GE(dp_rank, 0) << "invalid dp rank " << dp_rank;
  CHECK_LT(dp_rank, static_cast<int64_t>(token_nums.size()))
      << "dp rank " << dp_rank << " exceeds dp_global_token_nums size "
      << token_nums.size();

  const int32_t local_token_num = token_nums[dp_rank];
  CHECK_GT(local_token_num, 0)
      << "local dp token num must be positive for row-level conversion";
  CHECK_EQ(hidden_rows % local_token_num, 0)
      << "hidden rows " << hidden_rows
      << " must be divisible by local dp token num " << local_token_num;
  const int64_t row_factor = hidden_rows / local_token_num;

  std::vector<int32_t> row_token_nums;
  row_token_nums.reserve(token_nums.size());
  for (int32_t token_num : token_nums) {
    CHECK_GE(token_num, 0) << "dp token num must be non-negative";
    const int64_t row_token_num = static_cast<int64_t>(token_num) * row_factor;
    CHECK_LE(row_token_num, std::numeric_limits<int32_t>::max())
        << "row-level dp token num exceeds int32_t";
    row_token_nums.emplace_back(static_cast<int32_t>(row_token_num));
  }
  return row_token_nums;
}

torch::Tensor DeepseekV4SparseMoEBlockImpl::forward_selected(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params) {
  CHECK_GT(hidden_states.dim(), 1)
      << "hidden_states must have rows and hidden dimension";
  std::vector<int64_t> hidden_shape = hidden_states.sizes().vec();
  torch::Tensor hidden_rows =
      hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
  const int64_t row_count = hidden_rows.size(0);
  torch::Tensor topk_weights_2d =
      reshape_topk(topk_weights, row_count, "topk_weights");
  torch::Tensor topk_ids_2d = reshape_topk(topk_ids, row_count, "topk_ids");
  CHECK_EQ(topk_weights_2d.size(1), topk_ids_2d.size(1))
      << "topk_weights and topk_ids topk mismatch";

  std::vector<int32_t> row_token_nums;
  if (need_gather()) {
    row_token_nums = get_row_dp_tokens(row_count, input_params);
  }

  SelectedMoeInputs moe_inputs = gather_selected_moe_inputs(hidden_rows,
                                                            topk_weights_2d,
                                                            topk_ids_2d,
                                                            row_token_nums,
                                                            parallel_args_);

  torch::Tensor shared_out = moe_->forward_shared(moe_inputs.hidden_states);
  torch::Tensor routed_out = moe_->forward_selected(
      moe_inputs.hidden_states, moe_inputs.topk_weights, moe_inputs.topk_ids);
  ProcessGroup* reduce_group = routed_pg();
  CHECK(reduce_group != nullptr) << "routed process group is not initialized";
  if (reduce_group->world_size() > 1) {
    routed_out = parallel_state::reduce(routed_out, reduce_group);
  }

  torch::Tensor output = std::move(routed_out);
  if (shared_out.defined()) {
    output = output + shared_out;
  }
  if (moe_inputs.need_slice) {
    output = slice_selected_moe_output(
        std::move(output), row_token_nums, parallel_args_);
  }
  return output.reshape(hidden_shape);
}

}  // namespace layer
}  // namespace xllm
