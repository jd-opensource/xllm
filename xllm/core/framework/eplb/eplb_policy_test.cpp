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
