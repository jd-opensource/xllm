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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

namespace {

void set_death_style() { GTEST_FLAG_SET(death_test_style, "threadsafe"); }

InstanceInfo make_info(int32_t dp_size,
                       const std::vector<uint64_t>& cluster_ids) {
  InstanceInfo info;
  info.dp_size = dp_size;
  info.cluster_ids = cluster_ids;
  return info;
}

TEST(PdTopologyGuardTest, HomoTopoBypass) {
  const InstanceInfo info = make_info(2, {0, 1, 2, 3});

  const PdTopo topo = get_pd_topo(info);
  EXPECT_EQ(topo.dp_size, 2);
  EXPECT_EQ(topo.tp_size, 2);

  const PdTopoRule rule = check_mlu_pd_topo(topo, topo, false, "PULL", false);
  EXPECT_TRUE(rule.allow);
  EXPECT_FALSE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_TRUE(rule.reason.empty());
}

TEST(PdTopologyGuardTest, TryGetPdTopoReturnTopo) {
  const InstanceInfo info = make_info(2, {0, 1, 2, 3});

  PdTopo topo;
  std::string reason;
  EXPECT_TRUE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(topo.dp_size, 2);
  EXPECT_EQ(topo.tp_size, 2);
  EXPECT_TRUE(reason.empty());
}

TEST(PdTopologyGuardTest, HeteroTopoNeedMla) {
  const PdTopo local_topo{2, 2};
  const PdTopo remote_topo{1, 4};

  const PdTopoRule rule =
      check_mlu_pd_topo(local_topo, remote_topo, true, "PUSH", false);
  EXPECT_FALSE(rule.allow);
  EXPECT_TRUE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_EQ(rule.reason, "hetero pd requires enable_mla=true");
}

TEST(PdTopologyGuardTest, HeteroTopoNeedMluBuild) {
  const PdTopo local_topo{2, 2};
  const PdTopo remote_topo{1, 4};

  const PdTopoRule rule =
      check_mlu_pd_topo(local_topo, remote_topo, false, "PUSH", true);
  EXPECT_FALSE(rule.allow);
  EXPECT_TRUE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_EQ(rule.reason, "hetero pd requires is_mlu_build=true");
}

TEST(PdTopologyGuardTest, HeteroTopoNeedPushKv) {
  const PdTopo local_topo{2, 2};
  const PdTopo remote_topo{1, 4};

  const PdTopoRule rule =
      check_mlu_pd_topo(local_topo, remote_topo, true, "PULL", true);
  EXPECT_FALSE(rule.allow);
  EXPECT_TRUE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_EQ(rule.reason, "hetero pd requires kv_mode=PUSH");
}

TEST(PdTopologyGuardTest, HeteroTopoAllowOnMluPushMla) {
  const PdTopo local_topo{2, 2};
  const PdTopo remote_topo{1, 4};

  const PdTopoRule rule =
      check_mlu_pd_topo(local_topo, remote_topo, true, "PUSH", true);
  EXPECT_TRUE(rule.allow);
  EXPECT_TRUE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_TRUE(rule.reason.empty());
}

TEST(PdTopologyGuardTest, CheckPdRuleRejectInvalidLocalTopo) {
  const InstanceInfo local_info = make_info(0, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(1, {0, 1, 2, 3});

  const PdTopoRule rule =
      check_pd_rule(local_info, remote_info, true, "PUSH", true);
  EXPECT_FALSE(rule.allow);
  EXPECT_FALSE(rule.hetero);
  EXPECT_TRUE(rule.invalid_local);
  EXPECT_FALSE(rule.invalid_remote);
  EXPECT_EQ(rule.reason,
            "invalid local pd topo: dp_size must be greater than 0");
}

TEST(PdTopologyGuardTest, CheckPdRuleRejectInvalidRemoteTopo) {
  const InstanceInfo local_info = make_info(1, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(2, {0, 1, 2});

  const PdTopoRule rule =
      check_pd_rule(local_info, remote_info, true, "PUSH", true);
  EXPECT_FALSE(rule.allow);
  EXPECT_FALSE(rule.hetero);
  EXPECT_FALSE(rule.invalid_local);
  EXPECT_TRUE(rule.invalid_remote);
  EXPECT_EQ(rule.reason,
            "invalid remote pd topo: cluster_ids.size() must be divisible by "
            "dp_size");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectBadClusterSplit) {
  const InstanceInfo info = make_info(2, {0, 1, 2});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "cluster_ids.size() must be divisible by dp_size");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectEmptyClusterIds) {
  const InstanceInfo info = make_info(2, {});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "cluster_ids must not be empty");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectZeroDpSize) {
  const InstanceInfo info = make_info(0, {0, 1, 2, 3});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "dp_size must be greater than 0");
}

TEST(PdTopologyGuardTest, GetPdTopoRejectBadClusterSplit) {
  set_death_style();
  const InstanceInfo info = make_info(2, {0, 1, 2});

  EXPECT_DEATH(get_pd_topo(info),
               "cluster_ids.size\\(\\) must be divisible by dp_size");
}

TEST(PdTopologyGuardTest, GetPdTopoRejectEmptyClusterIds) {
  set_death_style();
  const InstanceInfo info = make_info(2, {});

  EXPECT_DEATH(get_pd_topo(info), "cluster_ids must not be empty");
}

}  // namespace

}  // namespace xllm
