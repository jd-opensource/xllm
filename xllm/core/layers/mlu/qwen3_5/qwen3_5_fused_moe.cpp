/* Copyright 2025-2026 The xLLM Authors.

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

#include "qwen3_5_fused_moe.h"

#include <glog/logging.h>

#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {
namespace {
torch::Tensor get_tensor_with_weight_suffix(const StateDict& state_dict,
                                            const std::string& tensor_name) {
  auto tensor = state_dict.get_tensor(tensor_name);
  if (!tensor.defined()) {
    tensor = state_dict.get_tensor(tensor_name + ".weight");
  }
  return tensor;
}

torch::Tensor slice_expert_weights(const torch::Tensor& weight,
                                   int64_t start_expert_id,
                                   int64_t num_experts_per_rank) {
  return weight
      .slice(0, start_expert_id, start_expert_id + num_experts_per_rank)
      .contiguous();
}

torch::Tensor shard_fused_gate_up(const torch::Tensor& fused_gate_up,
                                  int64_t rank,
                                  int64_t world_size) {
  if (world_size <= 1) {
    return fused_gate_up;
  }

  CHECK_GE(fused_gate_up.dim(), 2)
      << "gate_up_proj must have at least 2 dims, got "
      << fused_gate_up.sizes();
  CHECK_EQ(fused_gate_up.size(1) % 2, 0)
      << "gate_up_proj dim1 must be even, got " << fused_gate_up.size(1);
  const int64_t full_intermediate = fused_gate_up.size(1) / 2;
  CHECK_EQ(full_intermediate % world_size, 0)
      << "gate_up_proj intermediate dim is not divisible by world_size";
  const int64_t inter_shard = full_intermediate / world_size;

  torch::Tensor gate_full = fused_gate_up.slice(1, 0, full_intermediate);
  torch::Tensor up_full =
      fused_gate_up.slice(1, full_intermediate, full_intermediate * 2);
  torch::Tensor gate_shard =
      gate_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
  torch::Tensor up_shard =
      up_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
  return torch::cat({gate_shard, up_shard}, 1);
}

torch::Tensor shard_last_dim(const torch::Tensor& tensor,
                             int64_t rank,
                             int64_t world_size,
                             int64_t local_size) {
  if (world_size <= 1) {
    return tensor;
  }
  CHECK_EQ(tensor.size(-1), local_size * world_size)
      << "last dim is not divisible by world_size, tensor shape="
      << tensor.sizes() << ", local_size=" << local_size
      << ", world_size=" << world_size;
  return tensor.slice(
      tensor.dim() - 1, rank * local_size, (rank + 1) * local_size);
}

void copy_expanded_vector(const torch::Tensor& src, torch::Tensor& dst) {
  torch::Tensor reshaped = src.reshape({1, -1}).expand(dst.sizes());
  dst.copy_(reshaped);
}

bool load_fused_gate_up_sq(const StateDict& state_dict,
                           int64_t rank,
                           int64_t world_size,
                           int64_t start_expert_id,
                           int64_t num_experts_per_rank,
                           torch::Tensor& w13,
                           torch::Tensor& w13_scale,
                           torch::Tensor& input_smooth,
                           bool& w13_is_loaded,
                           bool& w13_scale_is_loaded,
                           bool& input_smooth_is_loaded) {
  torch::Tensor fused_qweight = state_dict.get_tensor("gate_up_proj.qweight");
  torch::Tensor fused_scale =
      state_dict.get_tensor("gate_up_proj.per_channel_scale");
  torch::Tensor fused_smooth = state_dict.get_tensor("gate_up_proj.smooth");
  if (!fused_qweight.defined() || !fused_scale.defined() ||
      !fused_smooth.defined()) {
    return false;
  }

  torch::Tensor qweight_shard =
      slice_expert_weights(shard_fused_gate_up(fused_qweight, rank, world_size),
                           start_expert_id,
                           num_experts_per_rank);
  CHECK_EQ(w13.sizes(), qweight_shard.sizes())
      << "qweight size mismatch for " << state_dict.prefix()
      << "gate_up_proj.qweight";
  w13.copy_(qweight_shard);
  w13_is_loaded = true;

  torch::Tensor scale_shard =
      slice_expert_weights(shard_fused_gate_up(fused_scale, rank, world_size),
                           start_expert_id,
                           num_experts_per_rank);
  CHECK_EQ(w13_scale.sizes(), scale_shard.sizes())
      << "per_channel_scale size mismatch for " << state_dict.prefix()
      << "gate_up_proj.per_channel_scale";
  w13_scale.copy_(scale_shard);
  w13_scale_is_loaded = true;

  CHECK_EQ(input_smooth.size(1), fused_smooth.numel())
      << "smooth size mismatch for " << state_dict.prefix()
      << "gate_up_proj.smooth";
  copy_expanded_vector(fused_smooth, input_smooth);
  input_smooth_is_loaded = true;
  return true;
}

bool load_fused_down_sq(const StateDict& state_dict,
                        int64_t rank,
                        int64_t world_size,
                        int64_t start_expert_id,
                        int64_t num_experts_per_rank,
                        torch::Tensor& w2,
                        torch::Tensor& w2_scale,
                        torch::Tensor& act_smooth,
                        bool& w2_is_loaded,
                        bool& w2_scale_is_loaded,
                        bool& act_smooth_is_loaded) {
  torch::Tensor fused_qweight = state_dict.get_tensor("down_proj.qweight");
  torch::Tensor fused_scale =
      state_dict.get_tensor("down_proj.per_channel_scale");
  torch::Tensor fused_smooth = state_dict.get_tensor("down_proj.smooth");
  if (!fused_qweight.defined() || !fused_scale.defined() ||
      !fused_smooth.defined()) {
    return false;
  }

  torch::Tensor qweight_shard = slice_expert_weights(
      shard_last_dim(fused_qweight, rank, world_size, w2.size(-1)),
      start_expert_id,
      num_experts_per_rank);
  CHECK_EQ(w2.sizes(), qweight_shard.sizes())
      << "qweight size mismatch for " << state_dict.prefix()
      << "down_proj.qweight";
  w2.copy_(qweight_shard);
  w2_is_loaded = true;

  torch::Tensor scale = fused_scale;
  if (fused_scale.dim() == w2_scale.dim() && fused_scale.dim() == 3) {
    scale = shard_last_dim(fused_scale, rank, world_size, w2_scale.size(-1));
  }
  torch::Tensor scale_shard =
      slice_expert_weights(scale, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w2_scale.sizes(), scale_shard.sizes())
      << "per_channel_scale size mismatch for " << state_dict.prefix()
      << "down_proj.per_channel_scale";
  w2_scale.copy_(scale_shard);
  w2_scale_is_loaded = true;

  torch::Tensor smooth_shard =
      shard_last_dim(fused_smooth, rank, world_size, act_smooth.size(-1));
  copy_expanded_vector(smooth_shard, act_smooth);
  act_smooth_is_loaded = true;
  return true;
}

bool load_fused_gate_up_fallback(const StateDict& state_dict,
                                 int64_t rank,
                                 int64_t world_size,
                                 int64_t start_expert_id,
                                 int64_t num_experts_per_rank,
                                 torch::Tensor& w13) {
  auto fused_gate_up =
      get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
  if (!fused_gate_up.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_gate_up.size(1) % 2, 0)
        << "gate_up_proj dim1 must be even, got " << fused_gate_up.size(1);
    const int64_t full_intermediate = fused_gate_up.size(1) / 2;
    CHECK_EQ(full_intermediate % world_size, 0)
        << "gate_up_proj intermediate dim is not divisible by world_size";
    const int64_t inter_shard = full_intermediate / world_size;

    auto gate_full = fused_gate_up.slice(1, 0, full_intermediate);
    auto up_full =
        fused_gate_up.slice(1, full_intermediate, full_intermediate * 2);
    auto gate_shard =
        gate_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    auto up_shard =
        up_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    fused_gate_up = torch::cat({gate_shard, up_shard}, 1);
  }

  auto gate_up_slice = slice_expert_weights(
      fused_gate_up, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w13.sizes(), gate_up_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.gate_up_proj";
  w13.copy_(gate_up_slice);
  return true;
}

bool load_fused_down_fallback(const StateDict& state_dict,
                              int64_t rank,
                              int64_t world_size,
                              int64_t start_expert_id,
                              int64_t num_experts_per_rank,
                              torch::Tensor& w2) {
  auto fused_down = get_tensor_with_weight_suffix(state_dict, "down_proj");
  if (!fused_down.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_down.size(2) % world_size, 0)
        << "down_proj dim2 is not divisible by world_size";
    const int64_t down_shard = fused_down.size(2) / world_size;
    fused_down =
        fused_down.slice(2, rank * down_shard, (rank + 1) * down_shard);
  }

  auto down_slice =
      slice_expert_weights(fused_down, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w2.sizes(), down_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.down_proj";
  w2.copy_(down_slice);
  return true;
}
}  // namespace

Qwen3_5FusedMoEImpl::Qwen3_5FusedMoEImpl(const ModelArgs& model_args,
                                         const FusedMoEArgs& moe_args,
                                         const QuantArgs& quant_args,
                                         const ParallelArgs& parallel_args,
                                         const torch::TensorOptions& options)
    : FusedMoEImpl(model_args, moe_args, quant_args, parallel_args, options) {
  if (n_shared_experts_ > 0) {
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
  }
}

void Qwen3_5FusedMoEImpl::load_experts(const StateDict& state_dict) {
  FusedMoEImpl::load_experts(state_dict);

  if (is_smoothquant_) {
    load_fused_gate_up_sq(state_dict,
                          tp_pg_->rank(),
                          tp_pg_->world_size(),
                          start_expert_id_,
                          num_experts_per_rank_,
                          w13_,
                          w13_scale_,
                          input_smooth_,
                          w13_is_loaded_,
                          w13_scale_is_loaded_,
                          input_smooth_is_loaded_);
    load_fused_down_sq(state_dict,
                       tp_pg_->rank(),
                       tp_pg_->world_size(),
                       start_expert_id_,
                       num_experts_per_rank_,
                       w2_,
                       w2_scale_,
                       act_smooth_,
                       w2_is_loaded_,
                       w2_scale_is_loaded_,
                       act_smooth_is_loaded_);
  } else {
    if (!w13_is_loaded_) {
      w13_is_loaded_ = load_fused_gate_up_fallback(state_dict,
                                                   tp_pg_->rank(),
                                                   tp_pg_->world_size(),
                                                   start_expert_id_,
                                                   num_experts_per_rank_,
                                                   w13_);
    }

    if (!w2_is_loaded_) {
      w2_is_loaded_ = load_fused_down_fallback(state_dict,
                                               tp_pg_->rank(),
                                               tp_pg_->world_size(),
                                               start_expert_id_,
                                               num_experts_per_rank_,
                                               w2_);
    }
  }
}

void Qwen3_5FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert."));
    auto weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      shared_expert_gate_->weight.data().copy_(weight);
    }
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

void Qwen3_5FusedMoEImpl::final_comm_allreduce(
    torch::Tensor& final_hidden_states,
    const torch::Tensor& hidden_states,
    torch::Tensor& shared_expert_output) {
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }
    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
  }

  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
    shared_expert_output = shared_experts_(hidden_states);
    if (shared_expert_gate_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(hidden_states));
      shared_expert_output = gate * shared_expert_output;
    }
    shared_expert_output =
        shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
  }

  // join for parallelization
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
    final_hidden_states += shared_expert_output;
  }
}

}  // namespace layer
}  // namespace xllm
