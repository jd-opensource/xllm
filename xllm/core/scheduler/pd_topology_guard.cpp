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

#include "scheduler/pd_topology_guard.h"

#include <glog/logging.h>

#include <cstddef>
#include <limits>

namespace xllm {

namespace {

bool fail_topo(const std::string& msg, std::string* reason) {
  if (reason != nullptr) {
    *reason = msg;
  }
  return false;
}

}  // namespace

bool try_get_pd_topo(const InstanceInfo& info,
                     PdTopo* topo,
                     std::string* reason) {
  if (topo == nullptr) {
    return fail_topo("topo must not be null", reason);
  }
  if (info.dp_size <= 0) {
    return fail_topo("dp_size must be greater than 0", reason);
  }

  const size_t cluster_num = info.cluster_ids.size();
  if (cluster_num == static_cast<size_t>(0)) {
    return fail_topo("cluster_ids must not be empty", reason);
  }

  const size_t dp_size = static_cast<size_t>(info.dp_size);
  if (cluster_num % dp_size != 0) {
    return fail_topo("cluster_ids.size() must be divisible by dp_size", reason);
  }
  if (cluster_num > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return fail_topo("cluster_ids.size() exceeds int32_t range", reason);
  }

  topo->dp_size = info.dp_size;
  topo->tp_size = static_cast<int32_t>(cluster_num / dp_size);
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

PdTopo get_pd_topo(const InstanceInfo& info) {
  PdTopo topo;
  std::string reason;
  CHECK(try_get_pd_topo(info, &topo, &reason)) << reason;
  return topo;
}

PdTopoRule check_pd_rule(const InstanceInfo& local,
                         const InstanceInfo& remote,
                         bool is_mlu_build,
                         const std::string& kv_mode,
                         bool enable_mla) {
  PdTopo local_topo;
  std::string reason;
  if (!try_get_pd_topo(local, &local_topo, &reason)) {
    return PdTopoRule{
        false, false, true, false, "invalid local pd topo: " + reason};
  }

  PdTopo remote_topo;
  if (!try_get_pd_topo(remote, &remote_topo, &reason)) {
    return PdTopoRule{
        false, false, false, true, "invalid remote pd topo: " + reason};
  }

  return check_mlu_pd_topo(
      local_topo, remote_topo, is_mlu_build, kv_mode, enable_mla);
}

PdTopoRule check_mlu_pd_topo(const PdTopo& local_topo,
                             const PdTopo& remote_topo,
                             bool is_mlu_build,
                             const std::string& kv_mode,
                             bool enable_mla) {
  const bool same_dp = local_topo.dp_size == remote_topo.dp_size;
  const bool same_tp = local_topo.tp_size == remote_topo.tp_size;
  if (same_dp && same_tp) {
    return PdTopoRule{true, false, false, false, ""};
  }

  if (!is_mlu_build) {
    return PdTopoRule{
        false, true, false, false, "hetero pd requires is_mlu_build=true"};
  }
  if (kv_mode != "PUSH") {
    return PdTopoRule{
        false, true, false, false, "hetero pd requires kv_mode=PUSH"};
  }
  if (!enable_mla) {
    return PdTopoRule{
        false, true, false, false, "hetero pd requires enable_mla=true"};
  }

  return PdTopoRule{true, true, false, false, ""};
}

}  // namespace xllm
