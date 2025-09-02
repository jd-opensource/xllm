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

#include "mapping_npu.h"

#include <glog/logging.h>

#define MAX_LCCL_COMM_DOMAIN 63
#define ENV_enable_extra_o_proj_tp false
#define ENV_lm_head_local_tp false
#define ENV_hccl_moe_ep_buffer 512
#define ENV_hccl_moe_tp_buffer 64
#define ENV_enable_dp_partition_up false

namespace xllm {
MappingNPU::MappingNPU(const std::string rank_table_file,
                       const int32_t world_size,
                       const int32_t rank,
                       const Options& options)
    : rank_table_file_(rank_table_file), rank_(rank), options_(options) {
  num_nodes_ = get_num_nodes();
  world_size_ = world_size;
  local_world_size_ = world_size / num_nodes_;
  attn_o_proj_tp_.backend("lccl");
  attn_inner_sp_.backend("lccl");
  parse_parallel_info();
  validate();
  get_tp_group(word_embed_tp_);
  get_dp_group(word_embed_dp_);
  get_tp_group(attn_tp_);
  get_tp_group(attn_o_proj_tp_);
  get_dp_group(attn_dp_);
  get_dp_group(attn_o_proj_dp_);
  get_tp_group(mlp_tp_);
  get_dp_group(mlp_dp_);
  get_tp_group(moe_tp_);
  get_dp_group(moe_ep_);
  get_tp_group(lm_head_tp_);
  get_dp_group(lm_head_dp_);
  get_tp_group(attn_inner_sp_);
  get_tp_group(attn_cp_);

  attn_cp_.group_size_ = 1;
  // o_proj mixture of tp and dp
  if (ENV_enable_extra_o_proj_tp) {
    get_domain(attn_o_proj_tp_, attn_o_proj_dp_, 0);
    get_domain(attn_tp_, attn_dp_, attn_o_proj_tp_.group_size());
    get_domain(attn_dp_,
               attn_tp_,
               attn_o_proj_tp_.group_size() + attn_dp_.group_size());
  } else if (ENV_lm_head_local_tp) {
    get_domain(lm_head_tp_, lm_head_dp_, 0);
    get_domain(attn_tp_, attn_dp_, attn_o_proj_tp_.group_size());
    get_domain(attn_dp_,
               attn_tp_,
               attn_o_proj_tp_.group_size() + attn_dp_.group_size());
  } else {
    get_domain(attn_tp_, attn_dp_, 0);
    get_domain(attn_dp_, attn_tp_, attn_dp_.group_size());
  }
  get_domain(moe_tp_, moe_ep_, 2 * world_size_);
  get_domain(moe_ep_, moe_tp_, 2 * world_size_ + moe_ep_.group_size());
  int32_t num_lccl_comm_shards = options_.num_lccl_comm_shards();
  int32_t num_lccl_per_shards =
      (MAX_LCCL_COMM_DOMAIN + 1) / num_lccl_comm_shards;
  int32_t lccl_comm_shard_id = options_.lccl_comm_shard_id();
  lccl_comm_domain_lower_bound_ = num_lccl_per_shards * lccl_comm_shard_id;
  lccl_comm_domain_upper_bound_ =
      num_lccl_per_shards * (lccl_comm_shard_id + 1);

  mlp_tp_.domain(std::to_string(MAX_LCCL_COMM_DOMAIN));
  moe_ep_.buffer_size(ENV_hccl_moe_ep_buffer);
  moe_tp_.buffer_size(ENV_hccl_moe_tp_buffer);
  attn_inner_sp_.domain(attn_tp_.domain());
}

int32_t MappingNPU::get_num_nodes() {
  if (rank_table_file_.empty()) {
    return 1;
  }
  if (std::filesystem::exists(rank_table_file_)) {
    JsonReader rank_table_reader;
    if (!rank_table_reader.parse(rank_table_file_)) {
      LOG(ERROR) << "Failed to parse rank table file: " << rank_table_file_;
      return 1;
    }
    if (auto data = rank_table_reader.value<std::string>("server_count")) {
      const std::string& str = data.value();
      if (str.empty()) return 1;
      size_t pos;
      int value = std::stoi(str, &pos);
      return value;
    }
  }
  return 1;
}

void MappingNPU::parse_parallel_info() {
  if (options_.dp_size() != -1) {
    attn_dp_.group_size(options_.dp_size());
  }
  attn_tp_.group_size(world_size_);
  mlp_tp_.group_size(world_size_);
  // pp.tp.group_size = world_size
  if (ENV_lm_head_local_tp) {
    // lm_head tp min(8, world_size)
    lm_head_dp_.group_size(num_nodes_);
    lm_head_tp_.group_size(std::min(8, world_size_));
  }
  // pp.group_size = 1
  // # microbatch_size
  // pp.microbatch_size = kwargs.get(MICROBATCH_SIZE)
  moe_tp_.group_size(world_size_);
  if (options_.tp_size() != -1) {
    attn_tp_.group_size(options_.tp_size());
    moe_tp_.group_size(options_.tp_size());
    // pp.tp.group_size(options_.tp_size());
  }

  // moe_tp
  if (options_.moe_tp_size() != -1) {
    moe_tp_.group_size(options_.moe_tp_size());
  }
  // moe_ep
  if (options_.moe_ep_size() != -1) {
    moe_ep_.group_size(options_.moe_ep_size());
  }
  // pp
  // if kwargs.get(PP, -1) != -1:
  //     pp.group_size = kwargs.get(PP, pp.group_size)
  // sp
  if (options_.sp_size() != -1) {
    attn_inner_sp_.group_size(options_.sp_size());
  }
  // word embed
  word_embed_tp_ = ParallelInfo(attn_tp_);
  word_embed_dp_ = ParallelInfo(attn_dp_);
  // lm_head
  if (ENV_enable_dp_partition_up) {
    if (!ENV_lm_head_local_tp) {
      lm_head_tp_ = ParallelInfo(attn_tp_);
      lm_head_dp_ = ParallelInfo(attn_dp_);
    }
  } else {
    lm_head_tp_ = ParallelInfo(mlp_tp_);
    lm_head_dp_ = ParallelInfo(mlp_dp_);
  }
  // convert attn tp type
  if (ENV_enable_extra_o_proj_tp && attn_tp_.group_size() > 1) {
    attn_o_proj_tp_.group_size(attn_tp_.group_size());
    attn_o_proj_dp_.group_size(world_size_ / attn_o_proj_tp_.group_size());
    attn_tp_.group_size(1);
  }
}

void MappingNPU::validate() {
  CHECK(world_size_ % num_nodes_ == 0)
      << "World size should be multiple of the number of nodes. "
         "Please check `world_size` and `ranktablefile`.";

  CHECK(attn_tp_.group_size() * attn_dp_.group_size() == world_size_)
      << "World size must equal to attention's dp_size * attention's tp_size. "
         "Attention's tp_size is " +
             std::to_string(attn_tp_.group_size()) +
             ". "
             "Attention's dp_size is " +
             std::to_string(attn_dp_.group_size()) +
             ". "
             "World size is " +
             std::to_string(world_size_) +
             ". "
             "Please check `dp`, `tp` and `world_size`.";

  if (attn_tp_.group_size() != world_size_) {
    CHECK(attn_tp_.group_size() <= local_world_size_)
        << "Attention's tp_size should be no greater than local world size, "
           "or equal to world size. "
           "Attention's tp_size is " +
               std::to_string(attn_tp_.group_size()) +
               ". "
               "World size is " +
               std::to_string(world_size_) +
               ". "
               "Local world size is " +
               std::to_string(local_world_size_) +
               ". "
               "Please check `tp`, `world_size` and `ranktablefile`.";
  }

  if (moe_ep_.group_size() > 1) {
    CHECK(moe_ep_.group_size() * moe_tp_.group_size() == world_size_)
        << "World size must equal to MoE's ep_size * MoE's tp_size. "
           "MoE's tp_size is " +
               std::to_string(moe_tp_.group_size()) +
               ". "
               "MoE's dp_size is " +
               std::to_string(moe_ep_.group_size()) +
               ". "
               "World size is " +
               std::to_string(world_size_) +
               ". "
               "Please check `moe_tp`, `moe_ep` and `world_size`.";
    if (moe_tp_.group_size() != world_size_) {
      CHECK(moe_tp_.group_size() <= local_world_size_)
          << "MoE's tp_size should be no greater than local world size, or "
             "equal to world size. "
             "MoE's tp_size is " +
                 std::to_string(moe_tp_.group_size()) +
                 ". "
                 "World size is " +
                 std::to_string(world_size_) +
                 ". "
                 "Local world size is " +
                 std::to_string(local_world_size_) +
                 ". "
                 "Please check `moe_tp`, `world_size` and `ranktablefile`.";
    }
  } else {
    CHECK(moe_tp_.group_size() == world_size_)
        << "World size must equal to MoE's tp_size. "
           "MoE's tp_size is " +
               std::to_string(moe_tp_.group_size()) + ". World size is " +
               std::to_string(world_size_) +
               ". Please check `tp`, `moe_tp` and `world_size`.";
  }
  if (attn_inner_sp_.group_size() > 1) {
    CHECK(attn_inner_sp_.group_size() == attn_tp_.group_size())
        << "Attention's sp_size must equal to attention's tp_size. "
           "Attention's sp_size is " +
               std::to_string(attn_inner_sp_.group_size()) +
               ". Attention's tp_size is " +
               std::to_string(attn_tp_.group_size()) +
               ". Please check `attn_sp` and `attn_tp`.";
  }
}

void MappingNPU::get_tp_group(ParallelInfo& parallel_info) {
  parallel_info.num_group(world_size_ / parallel_info.group_size());
  std::vector<std::vector<int32_t>> rank_per_group =
      parallel_info.rank_per_group();
  for (int i = 0; i <= parallel_info.num_group(); i++) {
    std::vector<int32_t> ranks;
    int32_t start = i * parallel_info.group_size();
    int32_t end = (i + 1) * parallel_info.group_size();
    for (int32_t j = start; j < end; ++j) {
      ranks.push_back(j);
    }
    rank_per_group.emplace_back(ranks);
  }
  parallel_info.rank_per_group(rank_per_group);
  auto [current_group_id, local_rank] =
      get_current_group_id(rank_per_group, rank_);
  CHECK(current_group_id >= 0 && local_rank >= 0)
      << "Failed to get current group id : " << current_group_id
      << " local_rank " << local_rank;
  parallel_info.current_group_id(current_group_id);
  parallel_info.rank(local_rank);
}

void MappingNPU::get_dp_group(ParallelInfo& parallel_info) {
  parallel_info.num_group(world_size_ / parallel_info.group_size());
  std::vector<std::vector<int32_t>> rank_per_group =
      parallel_info.rank_per_group();
  for (int i = 0; i <= parallel_info.num_group(); i++) {
    std::vector<int32_t> ranks;
    for (int32_t j = i; j < world_size_; j += parallel_info.num_group()) {
      ranks.push_back(j);
    }
    rank_per_group.emplace_back(ranks);
  }
  parallel_info.rank_per_group(rank_per_group);
  auto [current_group_id, local_rank] =
      get_current_group_id(rank_per_group, rank_);
  CHECK(current_group_id >= 0 && local_rank >= 0)
      << "Failed to get current group id : " << current_group_id
      << " local_rank " << local_rank;
  parallel_info.current_group_id(current_group_id);
  parallel_info.rank(local_rank);
}

void MappingNPU::get_domain(ParallelInfo& src,
                            ParallelInfo& dst,
                            const int32_t start_idx) {
  int32_t current_idx = dst.rank();
  src.domain(std::to_string(start_idx + current_idx));
}

std::tuple<int32_t, int32_t> MappingNPU::get_current_group_id(
    const std::vector<std::vector<int32_t>>& rank_per_group,
    int32_t target_rank_id) {
  for (int32_t idx = 0; idx < rank_per_group.size(); ++idx) {
    const auto& group = rank_per_group[idx];
    auto it = std::find(group.begin(), group.end(), target_rank_id);
    if (it != group.end()) {
      return std::make_tuple(idx, std::distance(group.begin(), it));
    }
  }
  return std::make_tuple(-1, -1);
}

nlohmann::json MappingNPU::to_json() {
  nlohmann::json data, lmhead_tp, lmhead_dp;

  // 开启lmhead后数据以dp策略形式输出
  if (ENV_enable_dp_partition_up) {
    lmhead_tp = attn_tp_.to_json();
    lmhead_dp = attn_dp_.to_json();
  } else {
    lmhead_tp = mlp_tp_.to_json();
    lmhead_dp = mlp_dp_.to_json();
  }
  data["moeEpSize"] = options_.moe_ep_size();
  data["moeTpSize"] = options_.moe_tp_size();
  data["attnDpSize"] = options_.dp_size();
  data["attnTpSize"] = options_.tp_size();
  data["worldSize"] = world_size_;
  data["rank"] = rank_;
  data["rankTableFile"] = rank_table_file_;
  data["localWorldSize"] = local_world_size_;
  data["lcclCommDomainLowerBound"] = lccl_comm_domain_lower_bound_;
  data["lcclCommDomainUpperBound"] = lccl_comm_domain_upper_bound_;
  data["wordEmbedTp"] = word_embed_tp_.to_json();
  data["wordEmbedDp"] = word_embed_dp_.to_json();
  data["attnTp"] = attn_tp_.to_json();
  data["attnDp"] = attn_dp_.to_json();
  data["attnInnerSp"] = attn_inner_sp_.to_json();
  data["attnOProjTp"] = attn_o_proj_tp_.to_json();
  data["attnOProjDp"] = attn_o_proj_dp_.to_json();
  data["mlpTp"] = mlp_tp_.to_json();
  data["mlpDp"] = mlp_dp_.to_json();
  data["moeTp"] = moe_tp_.to_json();
  data["moeEp"] = moe_ep_.to_json();
  data["lmHeadTp"] = lmhead_tp;
  data["lmHeadDp"] = lmhead_dp;
  data["lcocAttnTp"] = attn_tp_.to_json();
  data["attnCp"] = attn_cp_.to_json();

  return data;
}

}  // namespace xllm
