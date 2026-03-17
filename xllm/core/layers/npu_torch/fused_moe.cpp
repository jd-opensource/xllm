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

#include "fused_moe.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "common/global_flags.h"
#include "framework/parallel_state/npu_process_group.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "util/env_var.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#endif

namespace {
class CompletedWork final : public c10d::Work {
 public:
  CompletedWork() { finish(); }
};

inline c10::intrusive_ptr<c10d::Work> make_completed_work() {
  return c10::make_intrusive<CompletedWork>();
}

torch::Tensor create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype = torch::ScalarType::BFloat16) {
  torch::TensorOptions target_options = a.options().dtype(dtype);
  if (b.dim() != 2) {
    return torch::empty({a.size(0), b.size(1)}, target_options);
  }
  return torch::empty({group_list.size(0), a.size(0), b.size(0)},
                      target_options);
}

bool should_limit_topk_by_group(int64_t num_total_experts,
                                int64_t num_expert_group,
                                int64_t topk_group) {
  return num_expert_group > 1 && topk_group > 0 &&
         topk_group < num_expert_group &&
         num_total_experts % num_expert_group == 0;
}

torch::Tensor mask_scores_by_selected_groups(const torch::Tensor& scores,
                                             int64_t num_expert_group,
                                             int64_t topk_group) {
  const int64_t experts_per_group = scores.size(-1) / num_expert_group;
  const int64_t group_score_topk = std::min<int64_t>(2, experts_per_group);
  auto grouped_scores =
      scores.view({scores.size(0), num_expert_group, experts_per_group});
  auto group_topk = std::get<0>(torch::topk(grouped_scores,
                                            group_score_topk,
                                            /*dim=*/-1,
                                            /*largest=*/true,
                                            /*sorted=*/false));
  auto group_scores = group_topk.sum(/*dim=*/-1);
  auto selected_groups = std::get<1>(torch::topk(group_scores,
                                                 topk_group,
                                                 /*dim=*/-1,
                                                 /*largest=*/true,
                                                 /*sorted=*/false));

  auto selected_mask = torch::zeros(group_scores.sizes(),
                                    group_scores.options().dtype(torch::kBool));
  selected_mask.scatter_(
      /*dim=*/1,
      selected_groups,
      torch::ones(selected_groups.sizes(), selected_mask.options()));

  auto expert_mask =
      selected_mask.unsqueeze(-1)
          .expand({scores.size(0), num_expert_group, experts_per_group})
          .reshape_as(scores);
  return scores.masked_fill(expert_mask.logical_not(), 0.0);
}

bool should_use_minimax_ep_reference_moe() {
  return xllm::util::get_bool_env("XLLM_MINIMAX_EP_MOE_REFERENCE", true);
}

bool should_compare_minimax_ep_reference_moe() {
  return xllm::util::get_bool_env("XLLM_MINIMAX_COMPARE_EP_MOE_REFERENCE",
                                  false);
}

#if defined(USE_NPU)
bool is_npu_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_privateuseone();
}

void check_npu_input(const torch::Tensor& input) {
  CHECK(is_npu_tensor(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}

torch::Tensor flatten_for_scatter_gather(std::vector<torch::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return torch::empty(sizes, t.options());
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  switch (input.scalar_type()) {
    case torch::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case torch::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case torch::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case torch::kLong:
      return HCCL_DATA_TYPE_INT64;
    case torch::kInt:
      return HCCL_DATA_TYPE_INT32;
    case torch::kChar:
      return HCCL_DATA_TYPE_INT8;
    case torch::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      LOG(FATAL) << "Unconvertible HCCL type " << input.scalar_type();
  }
  return HCCL_DATA_TYPE_RESERVED;
}

class ExternalHcclProcessGroup final : public xllm::ProcessGroup {
 public:
  ExternalHcclProcessGroup(int32_t rank,
                           int32_t world_size,
                           const torch::Device& device,
                           HcclComm comm)
      : ProcessGroup(rank, world_size, device),
        comm_(comm),
        comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
    CHECK(comm_ != nullptr) << "External HCCL process group requires comm";
  }

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    CHECK_EQ(input.device(), device())
        << "input should be on the same device as the process group";
    CHECK_EQ(outputs.size(), world_size())
        << "outputs should have the same size as world_size";
    check_npu_input(input);
    torch::DeviceGuard device_guard(device());

    torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);
    const auto count = input.numel();
    const auto data_type = to_hccl_data_type(input);
    auto compute_stream = c10_npu::getCurrentNPUStream();

    auto ready = std::make_shared<c10_npu::NPUEvent>();
    ready->record(compute_stream);
    ready->block(comm_stream_);

    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                               comm_stream_);
    c10_npu::NPUCachingAllocator::recordStream(
        flattened_output.storage().data_ptr(), comm_stream_);

    HCCLCHECK(HcclAllGather(input.data_ptr(),
                            flattened_output.data_ptr(),
                            count,
                            data_type,
                            comm_,
                            comm_stream_.stream()));

    auto done = std::make_shared<c10_npu::NPUEvent>();
    done->record(comm_stream_);
    done->block(compute_stream);

    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
      outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
    }
  }

  c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs) override {
    allgather(input, outputs);
    return make_completed_work();
  }

  void allreduce(torch::Tensor& input) override {
    CHECK_EQ(input.device(), device())
        << "input should be on the same device as the process group";
    check_npu_input(input);
    torch::DeviceGuard device_guard(device());

    const auto count = input.numel();
    const auto data_type = to_hccl_data_type(input);
    auto compute_stream = c10_npu::getCurrentNPUStream();

    auto ready = std::make_shared<c10_npu::NPUEvent>();
    ready->record(compute_stream);
    ready->block(comm_stream_);

    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                               comm_stream_);

    HCCLCHECK(HcclAllReduce(input.data_ptr(),
                            input.data_ptr(),
                            count,
                            data_type,
                            HCCL_REDUCE_SUM,
                            comm_,
                            comm_stream_.stream()));

    auto done = std::make_shared<c10_npu::NPUEvent>();
    done->record(comm_stream_);
    done->block(compute_stream);
  }

 private:
  HcclComm comm_ = nullptr;
  c10_npu::NPUStream comm_stream_;
};

class WorldHcclSubgroupProcessGroup final : public xllm::ProcessGroup {
 public:
  WorldHcclSubgroupProcessGroup(int32_t rank,
                                const std::vector<uint32_t>& rank_ids,
                                xllm::ProcessGroup* world_pg)
      : ProcessGroup(rank,
                     static_cast<int32_t>(rank_ids.size()),
                     world_pg->device()),
        member_ranks_(rank_ids.begin(), rank_ids.end()),
        world_pg_(world_pg) {
    CHECK(world_pg_ != nullptr)
        << "WorldHcclSubgroupProcessGroup requires world_pg";
    CHECK(!member_ranks_.empty())
        << "WorldHcclSubgroupProcessGroup requires subgroup ranks";
  }

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    CHECK_EQ(outputs.size(), world_size())
        << "outputs should have the same size as subgroup world_size";

    std::vector<torch::Tensor> world_outputs(world_pg_->world_size());
    for (int32_t i = 0; i < world_pg_->world_size(); ++i) {
      world_outputs[i] = torch::empty_like(input);
    }
    world_pg_->allgather(input, world_outputs);

    for (int32_t i = 0; i < world_size(); ++i) {
      outputs[i].copy_(world_outputs[member_ranks_[i]], /*non_blocking=*/true);
    }
  }

  c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs) override {
    allgather(input, outputs);
    return make_completed_work();
  }

  void allreduce(torch::Tensor& input) override {
    std::vector<torch::Tensor> world_outputs(world_pg_->world_size());
    for (int32_t i = 0; i < world_pg_->world_size(); ++i) {
      world_outputs[i] = torch::empty_like(input);
    }
    world_pg_->allgather(input, world_outputs);

    input.zero_();
    for (int32_t global_rank : member_ranks_) {
      input.add_(world_outputs[global_rank]);
    }
  }

 private:
  std::vector<int32_t> member_ranks_;
  xllm::ProcessGroup* world_pg_ = nullptr;
};

std::unique_ptr<xllm::ProcessGroup> create_external_process_group(
    const atb_speed::common::ParallelInfo& parallel_info,
    const torch::Device& device,
    HcclComm comm) {
  CHECK(!parallel_info.rankIds.empty())
      << "parallel_info.rankIds must not be empty";
  CHECK(comm != nullptr) << "failed to fetch external HCCL comm";
  return std::make_unique<ExternalHcclProcessGroup>(
      static_cast<int32_t>(parallel_info.rank),
      static_cast<int32_t>(parallel_info.rankIds.size()),
      device,
      comm);
}

bool has_same_group_membership(const atb_speed::common::ParallelInfo& lhs,
                               const atb_speed::common::ParallelInfo& rhs) {
  return lhs.rank == rhs.rank && lhs.rankIds == rhs.rankIds;
}

std::unique_ptr<xllm::ProcessGroup> try_create_external_process_group(
    const atb_speed::common::ParallelInfo& parallel_info,
    const torch::Device& device,
    const std::string& backend,
    std::string* comm_domain_out = nullptr) {
  if (parallel_info.rankIds.empty()) {
    return nullptr;
  }

  HcclComm comm = nullptr;
  std::string comm_domain;
  parallel_info.InitCommDomain(comm, comm_domain, backend);
  if (comm_domain_out != nullptr) {
    *comm_domain_out = comm_domain;
  }
  if (comm == nullptr) {
    return nullptr;
  }
  return create_external_process_group(parallel_info, device, comm);
}
#endif

}  // namespace

namespace xllm {
namespace npu_torch_layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const layer::FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      num_expert_group_(std::max<int64_t>(model_args.n_group(), 1)),
      topk_group_(model_args.topk_group() > 0
                      ? model_args.topk_group()
                      : std::max<int64_t>(model_args.n_group(), 1)),
      route_scale_(model_args.routed_scaling_factor()),
      hidden_size_(model_args.hidden_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      is_gated_(moe_args.is_gated),
      has_score_bias_(false),
      has_bias_(false),
      skip_bias_add_(false),
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      hidden_act_(model_args.hidden_act()),
      scoring_func_(model_args.scoring_func().empty()
                        ? "softmax"
                        : model_args.scoring_func()),
      is_smoothquant_(false),
      is_minimax_m2_(model_args.model_type() == "minimax_m2"),
      quant_args_(quant_args),
      dp_size_(parallel_args.dp_size()),
      ep_size_(parallel_args.ep_size()),
      use_minimax_ep_reference_(model_args.model_type() == "minimax_m2" &&
                                parallel_args.ep_size() > 1 &&
                                should_use_minimax_ep_reference_moe()),
      compare_minimax_ep_reference_(model_args.model_type() == "minimax_m2" &&
                                    parallel_args.ep_size() > 1 &&
                                    should_compare_minimax_ep_reference_moe()),
      options_(options),
      world_pg_(parallel_args.process_group_),
      tp_pg_(parallel_args.tp_group_),
      dp_local_pg_(parallel_args.dp_local_process_group_),
      moe_ep_pg_(parallel_args.moe_ep_group_) {
  if (use_minimax_ep_reference_) {
    LOG_FIRST_N(WARNING, 1)
        << "MiniMax EP MoE is using the correctness-first local reference "
           "path on NPU torch. This avoids the grouped-kernel EP path until "
           "its output matches the reference.";
  } else if (compare_minimax_ep_reference_) {
    LOG_FIRST_N(INFO, 1)
        << "MiniMax EP MoE grouped-kernel path will be compared against the "
           "local reference implementation on NPU torch.";
  }

  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  const std::string& topk_method = model_args.topk_method();
  int64_t ep_size = ep_size_;
  int64_t ep_rank = 0;
  CHECK(tp_pg_ != nullptr) << "FusedMoE requires tp_group_";
#if defined(USE_NPU)
  if (!parallel_args.mapping_data().empty() && dp_size_ > 1 && ep_size > 1) {
    const auto attn_dp_info =
        parallel_args.mapping().Get(atb_speed::base::ATTN_DP);
    const auto moe_ep_info =
        parallel_args.mapping().Get(atb_speed::base::MOE_EP);
    const bool same_dispatch_group =
        !attn_dp_info.rankIds.empty() && !moe_ep_info.rankIds.empty() &&
        has_same_group_membership(attn_dp_info, moe_ep_info);
    const HcclComm dispatch_and_combine_comm =
        parallel_args.dispatchAndCombineHcclComm();
    if (same_dispatch_group && dispatch_and_combine_comm != nullptr) {
      shared_dispatch_pg_ = create_external_process_group(
          attn_dp_info, options.device(), dispatch_and_combine_comm);
      dp_local_pg_ = shared_dispatch_pg_.get();
      moe_ep_pg_ = shared_dispatch_pg_.get();
      LOG_FIRST_N(INFO, 1)
          << "MiniMax FusedMoE reusing the ATB dispatch/combine HCCL comm for "
             "the shared ATTN_DP/MOE_EP subgroup"
          << " (attn_dp_buffer=" << attn_dp_info.bufferSize
          << ", moe_ep_buffer=" << moe_ep_info.bufferSize << ")";
    } else if (same_dispatch_group) {
      std::string comm_domain;
      shared_dispatch_pg_ =
          try_create_external_process_group(moe_ep_info,
                                            options.device(),
                                            FLAGS_communication_backend,
                                            &comm_domain);
      if (shared_dispatch_pg_ != nullptr) {
        dp_local_pg_ = shared_dispatch_pg_.get();
        moe_ep_pg_ = shared_dispatch_pg_.get();
        LOG_FIRST_N(INFO, 1)
            << "MiniMax FusedMoE lazily initialized the shared ATTN_DP/MOE_EP "
               "subgroup via MOE_EP ParallelInfo"
            << " (backend=" << FLAGS_communication_backend << ", comm_domain="
            << (comm_domain.empty() ? "<empty>" : comm_domain) << ")";
      } else {
        LOG_FIRST_N(WARNING, 1)
            << "MiniMax FusedMoE could not lazily initialize the shared "
               "ATTN_DP/MOE_EP subgroup via MOE_EP ParallelInfo"
            << " (backend=" << FLAGS_communication_backend << ")";
      }
      if (shared_dispatch_pg_ == nullptr && dp_local_pg_ != nullptr &&
          moe_ep_pg_ != nullptr) {
        LOG_FIRST_N(INFO, 1)
            << "MiniMax FusedMoE reusing the prebuilt HCCL subgroup process "
               "groups for the shared ATTN_DP/MOE_EP membership because no "
               "shared external dispatch/combine comm is available"
            << " (attn_dp_world_size=" << dp_local_pg_->world_size()
            << ", moe_ep_world_size=" << moe_ep_pg_->world_size()
            << ", attn_dp_ranks=" << attn_dp_info.rankIds.size()
            << ", moe_ep_ranks=" << moe_ep_info.rankIds.size() << ")";
      } else if (shared_dispatch_pg_ == nullptr && world_pg_ != nullptr) {
        // MiniMax's DP shards stay aligned across TP columns, so the live
        // world HCCL group can emulate this shared subgroup without lazily
        // bootstrapping a second communicator during warmup/capture.
        shared_dispatch_pg_ = std::make_unique<WorldHcclSubgroupProcessGroup>(
            static_cast<int32_t>(attn_dp_info.rank),
            attn_dp_info.rankIds,
            world_pg_);
        dp_local_pg_ = shared_dispatch_pg_.get();
        moe_ep_pg_ = shared_dispatch_pg_.get();
        LOG_FIRST_N(WARNING, 1)
            << "MiniMax FusedMoE emulating the shared ATTN_DP/MOE_EP subgroup "
               "on top of the existing world HCCL process group"
            << " (attn_dp_ranks=" << attn_dp_info.rankIds.size()
            << ", moe_ep_ranks=" << moe_ep_info.rankIds.size()
            << ", world_size=" << world_pg_->world_size() << ")";
      }
      if (shared_dispatch_pg_ == nullptr) {
        LOG_FIRST_N(WARNING, 1)
            << "MiniMax FusedMoE could not create a shared ATTN_DP/MOE_EP "
               "external HCCL subgroup and has no world-group emulation "
               "fallback; reusing the prebuilt subgroup process groups "
               "instead"
            << " (attn_dp_ranks=" << attn_dp_info.rankIds.size()
            << ", moe_ep_ranks=" << moe_ep_info.rankIds.size()
            << ", dp_local_pg=" << (dp_local_pg_ != nullptr ? "set" : "null")
            << ", moe_ep_pg=" << (moe_ep_pg_ != nullptr ? "set" : "null")
            << ")";
      }
    }

    if (same_dispatch_group && dp_local_pg_ != nullptr) {
      // The DP gather and the EP result-reduce land on the same rank set in
      // the current dp=2,ep=2 topology, so one comm wrapper is sufficient.
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "MiniMax FusedMoE falling back to torch subgroup process groups"
          << " (dispatch_comm="
          << (dispatch_and_combine_comm != nullptr ? "set" : "null")
          << ", attn_dp_ranks=" << attn_dp_info.rankIds.size()
          << ", moe_ep_ranks=" << moe_ep_info.rankIds.size()
          << ", same_group=" << same_dispatch_group << ")";
    }
  }
#endif
  if (ep_size > 1) {
    CHECK(moe_ep_pg_ != nullptr) << "FusedMoE requires moe_ep_group_";
    CHECK(tp_pg_ != nullptr)
        << "FusedMoE requires moe_tp_group_ when ep_size > 1";
    ep_rank = moe_ep_pg_->rank();
  }

  // smoothquant check: If quant_method is not empty, only w8a8 smoothquant is
  // supported
  if (!quant_args.quant_method().empty()) {
    if (quant_args.quant_method() != "smoothquant" || quant_args.bits() != 8 ||
        !quant_args.activation_dynamic()) {
      LOG(FATAL) << "FusedMoE only supports w8a8 smoothquant quantization when "
                    "quant_method is set. "
                 << "Got quant_method=" << quant_args.quant_method()
                 << ", bits=" << quant_args.bits()
                 << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
    // If confirmed as smoothquant w8a8, set is_smoothquant_ to true
    is_smoothquant_ = true;
  } else {
    is_smoothquant_ = false;
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias", torch::empty({num_experts}, options), false);
    has_score_bias_ = true;
  }

  gate_ = register_module(
      "gate_proj",
      layer::ReplicatedLinear(
          hidden_size_, num_experts, false, quant_args, options));
  if (n_shared_experts_ > 0) {
    /*
    The shared_experts are usually implemented using the RowParallelLinear
    layer. Typically, this output serves as the enable_result_reduction results
    for the module. If only tensor parallelism is applied, immediate
    reduction of the shared_experts output isn't necessary; instead, we perform
    the reduction once at the end of the MoE operation.
    */
    shared_experts_ =
        register_module("shared_experts",
                        layer::DenseMLP(hidden_size_,
                                        intermediate_size * n_shared_experts_,
                                        is_gated_,
                                        false,
                                        hidden_act_,
                                        /*enable_result_reduction=*/false,
                                        quant_args,
                                        tp_pg_,
                                        options));
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
  }

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;
  if (is_smoothquant_) {
    auto quant_option = options_.dtype(torch::kInt8);
    auto fp_option = options_.dtype(torch::kFloat32);
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            quant_option),
        false);
    w13_scale_ = register_parameter(
        "w13_scale",
        torch::empty({num_experts_per_rank_, local_intermediate_size * 2},
                     fp_option),
        false);
    input_smooth_ = register_parameter(
        "input_smooth",
        torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            quant_option),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
        false);
    act_smooth_ = register_parameter(
        "act_smooth",
        torch::empty({num_experts_per_rank_, local_intermediate_size},
                     fp_option),
        false);

  } else {
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            options_),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            options_),
        false);
  }
}

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  torch::Tensor routing_scores;
  if (scoring_func_ == "sigmoid") {
    routing_scores = torch::sigmoid(router_logits_2d.to(torch::kFloat32));
  } else if (scoring_func_ == "softmax") {
    routing_scores =
        torch::softmax(router_logits_2d.to(torch::kFloat32), /*dim=*/-1);
  } else {
    LOG(FATAL) << "Unsupported MoE scoring_func on NPU torch path: "
               << scoring_func_;
  }

  torch::Tensor choice_scores = routing_scores;
  if (e_score_correction_bias_.defined()) {
    choice_scores =
        choice_scores +
        e_score_correction_bias_.to(routing_scores.device(), torch::kFloat32);
  }

  if (should_limit_topk_by_group(
          num_total_experts_, num_expert_group_, topk_group_)) {
    choice_scores = mask_scores_by_selected_groups(
        choice_scores, num_expert_group_, topk_group_);
  }

  auto topk_result = torch::topk(choice_scores,
                                 topk_,
                                 /*dim=*/-1,
                                 /*largest=*/true,
                                 /*sorted=*/false);
  auto topk_ids = std::get<1>(topk_result).to(torch::kInt32).contiguous();
  auto topk_weights = routing_scores.gather(
      /*dim=*/1, topk_ids.to(torch::kLong).contiguous());
  if (renormalize_) {
    topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
  }
  if (route_scale_ != 1.0) {
    topk_weights = topk_weights * route_scale_;
  }
  topk_weights = topk_weights.contiguous();

  torch::Tensor routing_expert_idx = topk_ids;
  int64_t routing_expert_num = num_experts_per_rank_;
  std::array<int64_t, 2> active_expert_range = {0, num_experts_per_rank_};
  if (ep_size_ > 1) {
    // torch_npu's builtin moe_init_routing expects expert_idx and
    // active_expert_range in the same local expert-id space, with
    // active_expert_range[1] <= expert_num. MiniMax routes over the global
    // expert space first, so remap local experts onto [0,
    // num_experts_per_rank_) and send non-local experts to a sentinel just
    // outside the active range.
    auto local_expert_idx = topk_ids.to(torch::kInt64) - start_expert_id_;
    auto local_expert_mask =
        (local_expert_idx >= 0) & (local_expert_idx < num_experts_per_rank_);
    auto invalid_expert_idx =
        torch::full_like(local_expert_idx, num_experts_per_rank_);
    routing_expert_idx =
        torch::where(local_expert_mask, local_expert_idx, invalid_expert_idx)
            .to(torch::kInt32)
            .contiguous();
    routing_expert_num += 1;
  }

  xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
  moe_init_routing_params.x = hidden_states_2d;
  moe_init_routing_params.expert_idx = routing_expert_idx;
  moe_init_routing_params.scale = std::nullopt;
  moe_init_routing_params.offset = std::nullopt;
  moe_init_routing_params.active_num = hidden_states_2d.size(0) * topk_;
  moe_init_routing_params.expert_capacity = 0;
  moe_init_routing_params.expert_num = routing_expert_num;
  moe_init_routing_params.drop_pad_mode = 0;
  moe_init_routing_params.expert_tokens_num_type = 1;
  moe_init_routing_params.expert_tokens_num_flag = true;
  moe_init_routing_params.row_idx_type = 0;
  moe_init_routing_params.active_expert_range = torch::IntArrayRef(
      active_expert_range.data(), active_expert_range.size());
  moe_init_routing_params.quant_mode = -1;
  // TODO: NPU moe_init_routing_v2 is equivalent to moe_gen_idx +
  // moe_expand_input (and the token_count/cusum outputs) on other backends.
  auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] =
      xllm::kernel::moe_init_routing_v2(moe_init_routing_params);
  (void)dynamic_scale;

  // collect the selected tensor
  selected_expert_info.reduce_weight = topk_weights;
  selected_expert_info.combine_idx = expand_row_ids;
  selected_expert_info.token_count_slice = group_list;
  selected_expert_info.cusum_token_count = group_list;
  return expand_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_expert(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  // prepare the parameters for MoE computation
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});

  // Step 1-3: select experts
  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states =
      select_experts(hidden_states_2d, router_logits_2d, selected_expert_info);

  auto run_group_gemm = [&](const torch::Tensor& input,
                            const torch::Tensor& weight) -> torch::Tensor {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = input;
    group_gemm_params.b = weight;
    group_gemm_params.group_list = selected_expert_info.token_count_slice;
    group_gemm_params.split_item = 2;
    group_gemm_params.group_type = 0;
    group_gemm_params.group_list_type = 1;
    return xllm::kernel::group_gemm(group_gemm_params);
  };

  // Step 4: group gemm 1
  torch::Tensor w13_gemm = w13_;
  if (w13_gemm.size(1) != expand_hidden_states.size(1)) {
    w13_gemm = w13_gemm.transpose(1, 2).contiguous();
  }

  torch::Tensor act_out;
  if (is_minimax_m2_ && is_gated_ && hidden_act_ == "silu") {
    const int64_t local_intermediate_size = w13_gemm.size(2) / 2;
    auto gate_proj = run_group_gemm(
        expand_hidden_states,
        w13_gemm.slice(/*dim=*/2, /*start=*/0, /*end=*/local_intermediate_size)
            .contiguous());
    auto up_proj =
        run_group_gemm(expand_hidden_states,
                       w13_gemm
                           .slice(/*dim=*/2,
                                  /*start=*/local_intermediate_size,
                                  /*end=*/local_intermediate_size * 2)
                           .contiguous());
    act_out = torch::silu(gate_proj) * up_proj;
  } else {
    torch::Tensor gemm1_out =
        create_group_gemm_output(expand_hidden_states,
                                 w13_gemm,
                                 selected_expert_info.token_count_slice,
                                 hidden_states_dtype);
    gemm1_out = run_group_gemm(expand_hidden_states, w13_gemm);

    // Step 5: activation or scaled quantization(fused with activation)
    act_out = is_gated_
                  ? gemm1_out.slice(1, 0, gemm1_out.size(1) / 2).contiguous()
                  : gemm1_out;

    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    xllm::kernel::active(activation_params);
  }
  // Step 6: group gemm 2
  torch::Tensor w2_gemm = w2_;
  if (w2_gemm.size(1) != act_out.size(1)) {
    w2_gemm = w2_gemm.transpose(1, 2).contiguous();
  }
  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_gemm,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);
  gemm2_out = run_group_gemm(act_out, w2_gemm);

  // Step 7: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
  xllm::kernel::MoeCombineResultParams moe_combine_params;
  moe_combine_params.input = gemm2_out;
  moe_combine_params.reduce_weight = selected_expert_info.reduce_weight;
  moe_combine_params.gather_ids = selected_expert_info.combine_idx;
  final_hidden_states = xllm::kernel::moe_combine_result(moe_combine_params);
  if (shared_output.has_value()) {
    final_hidden_states = final_hidden_states + shared_output.value();
  }
  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  if (ep_size_ > 1) {
    final_hidden_states =
        parallel_state::reduce(final_hidden_states, moe_ep_pg_);
  }
  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_expert_reference(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  CHECK_EQ(hidden_act_, "silu")
      << "MiniMax EP MoE reference path only supports silu activation";
  CHECK(is_gated_) << "MiniMax EP MoE reference path expects gated experts";

  auto hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
  auto router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)}).contiguous();

  torch::Tensor routing_scores;
  if (scoring_func_ == "sigmoid") {
    routing_scores = torch::sigmoid(router_logits_2d.to(torch::kFloat32));
  } else if (scoring_func_ == "softmax") {
    routing_scores =
        torch::softmax(router_logits_2d.to(torch::kFloat32), /*dim=*/-1);
  } else {
    LOG(FATAL) << "Unsupported MoE scoring_func on NPU torch path: "
               << scoring_func_;
  }

  auto choice_scores = routing_scores;
  if (e_score_correction_bias_.defined()) {
    choice_scores =
        choice_scores +
        e_score_correction_bias_.to(routing_scores.device(), torch::kFloat32);
  }
  if (should_limit_topk_by_group(
          num_total_experts_, num_expert_group_, topk_group_)) {
    choice_scores = mask_scores_by_selected_groups(
        choice_scores, num_expert_group_, topk_group_);
  }

  auto topk_result = torch::topk(choice_scores,
                                 topk_,
                                 /*dim=*/-1,
                                 /*largest=*/true,
                                 /*sorted=*/false);
  auto topk_ids = std::get<1>(topk_result).to(torch::kLong).contiguous();
  auto topk_weights = routing_scores.gather(/*dim=*/1, topk_ids).contiguous();
  if (renormalize_) {
    topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
  }
  if (route_scale_ != 1.0) {
    topk_weights = topk_weights * route_scale_;
  }

  auto output = torch::zeros_like(hidden_states_2d);
  for (int64_t local_expert_idx = 0; local_expert_idx < num_experts_per_rank_;
       ++local_expert_idx) {
    const int64_t global_expert_idx = start_expert_id_ + local_expert_idx;
    auto expert_matches = (topk_ids == global_expert_idx).nonzero();
    if (expert_matches.numel() == 0) {
      continue;
    }

    auto token_idx =
        expert_matches.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
    auto topk_slot =
        expert_matches.select(/*dim=*/1, /*index=*/1).to(torch::kLong);
    auto current_states = hidden_states_2d.index_select(/*dim=*/0, token_idx);

    auto current_w13 = w13_.index({local_expert_idx});
    if (current_w13.size(0) != hidden_size_) {
      current_w13 = current_w13.transpose(/*dim0=*/0, /*dim1=*/1);
    }
    current_w13 = current_w13.contiguous();

    auto gate_up = torch::matmul(current_states, current_w13);
    const int64_t local_intermediate_size = gate_up.size(-1) / 2;
    auto gate_proj =
        gate_up.slice(/*dim=*/-1, /*start=*/0, /*end=*/local_intermediate_size);
    auto up_proj = gate_up.slice(/*dim=*/-1,
                                 /*start=*/local_intermediate_size,
                                 /*end=*/local_intermediate_size * 2);
    auto activated = torch::silu(gate_proj) * up_proj;

    auto current_w2 = w2_.index({local_expert_idx});
    if (current_w2.size(0) != local_intermediate_size) {
      current_w2 = current_w2.transpose(/*dim0=*/0, /*dim1=*/1);
    }
    current_w2 = current_w2.contiguous();

    auto expert_out = torch::matmul(activated, current_w2);
    auto combine_weight = topk_weights.index({token_idx, topk_slot})
                              .to(expert_out.scalar_type())
                              .unsqueeze(/*dim=*/-1);
    output.index_add_(/*dim=*/0, token_idx, expert_out * combine_weight);
  }

  if (shared_output.has_value()) {
    output = output + shared_output.value().reshape(
                          {-1, shared_output.value().size(-1)});
  }

  auto final_hidden_states = output.reshape(hidden_states.sizes());
  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  if (ep_size_ > 1) {
    final_hidden_states =
        parallel_state::reduce(final_hidden_states, moe_ep_pg_);
  }
  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  auto input = hidden_states;
  bool need_slice = false;
  if (dp_size_ > 1 && ep_size_ > 1) {
    CHECK(dp_local_pg_ != nullptr)
        << "FusedMoE requires dp_local_process_group_ for dp+ep";
    input = parallel_state::gather(
        input, dp_local_pg_, input_params.dp_global_token_nums);
    need_slice = true;
  }

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_(input);
    if (shared_expert_gate_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      if (shared_output.has_value()) {
        torch::Tensor res = gate * shared_output.value();
        shared_output = res;
      }
    }
  }
  auto router_logits = gate_(input);
  torch::Tensor output;
  torch::Tensor reference_output;
  torch::Tensor grouped_output;
  if (use_minimax_ep_reference_ || compare_minimax_ep_reference_) {
    reference_output =
        forward_expert_reference(input, router_logits, shared_output);
  }
  if (!use_minimax_ep_reference_ || compare_minimax_ep_reference_) {
    grouped_output = forward_expert(input, router_logits, shared_output);
  }
  if (compare_minimax_ep_reference_) {
    auto diff = (grouped_output.to(torch::kFloat32) -
                 reference_output.to(torch::kFloat32))
                    .abs();
    LOG_FIRST_N(INFO, 1)
        << "MiniMax EP MoE grouped path diff vs reference: max_abs="
        << diff.max().to(torch::kCPU).item<double>()
        << ", mean_abs=" << diff.mean().to(torch::kCPU).item<double>();
  }
  output = use_minimax_ep_reference_ ? reference_output : grouped_output;

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int64_t dp_rank = dp_local_pg_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
  }
  return output;
}

void FusedMoEImpl::load_e_score_correction_bias(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  if (is_smoothquant_) {
    LOAD_MOE_FUSED_WEIGHT("qweight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT("per_channel_scale", w1_scale, w3_scale, w13_scale);
    LOAD_MOE_WEIGHT("up_proj.", "smooth", input_smooth, -1);
    LOAD_MOE_WEIGHT("down_proj.", "qweight", w2, 1);
    LOAD_MOE_WEIGHT("down_proj.", "per_channel_scale", w2_scale, -1);
    LOAD_MOE_WEIGHT("down_proj.", "smooth", act_smooth, 0);
  } else {
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
  }
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
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
  load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace npu_torch_layer
}  // namespace xllm
