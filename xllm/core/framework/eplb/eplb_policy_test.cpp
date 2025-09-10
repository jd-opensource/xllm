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

#include "eplb_policy.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {

TEST(EplbPolicyTest, Build) {
  std::string rank_table_file;
  EplbPolicy eplb_policy(5, 4, 1);
  std::vector<torch::Tensor> tensors;
  tensors.push_back(torch::arange(0, 16));

  auto expert_load = torch::stack(tensors, 0);
  expert_load[0] =
      torch::tensor({100, 100, 100, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 100});
  auto [rebalance_expert, enable_update_vec] =
      eplb_policy.rebalance_experts(expert_load);
  LOG(INFO) << "rebalance_expert:" << rebalance_expert;
}

}  // namespace xllm
