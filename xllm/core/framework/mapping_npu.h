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

#pragma once

#include "core/common/macros.h"
#include "core/util/json_reader.h"

namespace xllm {

struct ParallelInfo {
  ParallelInfo(int32_t group_size = 1,
               int32_t num_group = -1,
               std::vector<std::vector<int32_t>> rank_per_group = {},
               int32_t current_group_id = {},
               int32_t rank = 0,
               const std::string& domain = "",
               int32_t buffer_size = 128,
               const std::string& backend = "")
      : group_size_(group_size),
        num_group_(num_group),
        rank_per_group_(rank_per_group),
        current_group_id_(current_group_id),
        rank_(rank),
        domain_(domain),
        buffer_size_(buffer_size),
        backend_(backend) {};

  ParallelInfo(const ParallelInfo& other)
      : group_size_(other.group_size()),
        num_group_(other.num_group()),
        rank_per_group_(other.rank_per_group()),
        current_group_id_(other.current_group_id()),
        rank_(other.rank()),
        domain_(other.domain()),
        buffer_size_(other.buffer_size()),
        backend_(other.backend()) {};

  nlohmann::json to_json() {
    nlohmann::json data;
    data["group_size"] = group_size_;
    // data["num_group"] = num_group_;
    data["rankIds"] = rank_per_group_[current_group_id_];
    data["groupId"] = current_group_id_;
    data["rank"] = rank_;
    // data["domain"] = domain_;
    data["bufferSize"] = buffer_size_;
    data["backend"] = backend_;
    return data;
  }

  // group size
  PROPERTY(int32_t, group_size) = 1;

  // size of current group
  PROPERTY(int32_t, num_group) = -1;

  // id of current group
  PROPERTY(int32_t, current_group_id) = -1;

  PROPERTY(std::vector<std::vector<int32_t>>, rank_per_group) = {};

  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // domain of current process
  PROPERTY(std::string, domain);

  // buffer size
  PROPERTY(int32_t, buffer_size) = 128;

  // backend : lccl / hccl
  PROPERTY(std::string, backend);
};

class MappingNPU final {
 public:
  struct Options {
    PROPERTY(int32_t, num_lccl_comm_shards) = 1;
    PROPERTY(int32_t, lccl_comm_shard_id) = 0;
    // dp size
    PROPERTY(int32_t, dp_size) = -1;
    // tp size
    PROPERTY(int32_t, tp_size) = -1;
    // moe tp size
    PROPERTY(int32_t, moe_tp_size) = -1;
    // moe ep size
    PROPERTY(int32_t, moe_ep_size) = -1;
    // pp size (dont support now)
    PROPERTY(int32_t, pp_size) = -1;
    // sp size (dont support now)
    PROPERTY(int32_t, sp_size) = -1;
  };

  MappingNPU(std::string rank_table_file,
             const int32_t world_size,
             const int32_t rank,
             const Options& options);

  int32_t get_num_nodes();

  void parse_parallel_info();

  void validate();

  void get_tp_group(ParallelInfo& parallel_info);

  void get_dp_group(ParallelInfo& parallel_info);

  void get_domain(ParallelInfo& src,
                  ParallelInfo& dst,
                  const int32_t start_idx);

  std::tuple<int32_t, int32_t> get_current_group_id(
      const std::vector<std::vector<int>>& rank_per_group,
      int target_rank_id);

  nlohmann::json to_json();

 private:
  Options options_;
  std::string rank_table_file_;
  int32_t num_nodes_;
  int32_t world_size_ = 0;
  int32_t rank_ = 0;
  int32_t local_world_size_ = 0;
  ParallelInfo word_embed_tp_ = ParallelInfo();
  ParallelInfo word_embed_dp_ = ParallelInfo();
  ParallelInfo attn_tp_ = ParallelInfo();
  ParallelInfo attn_o_proj_tp_ = ParallelInfo();
  ParallelInfo attn_dp_ = ParallelInfo();
  ParallelInfo attn_o_proj_dp_ = ParallelInfo();
  ParallelInfo mlp_tp_ = ParallelInfo();
  ParallelInfo mlp_dp_ = ParallelInfo();
  ParallelInfo moe_tp_ = ParallelInfo();
  ParallelInfo moe_ep_ = ParallelInfo();
  ParallelInfo lm_head_tp_ = ParallelInfo();
  ParallelInfo lm_head_dp_ = ParallelInfo();
  ParallelInfo attn_inner_sp_ = ParallelInfo();
  ParallelInfo attn_cp_ = ParallelInfo();

  int32_t lccl_comm_domain_lower_bound_;
  int32_t lccl_comm_domain_upper_bound_;
};
}  // namespace xllm
