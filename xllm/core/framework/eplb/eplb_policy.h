#pragma once

#include <torch/torch.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace xllm {

class EplbPolicy {
 public:
  EplbPolicy(int32_t device_experts_num, int32_t device_num, int32_t layer_num);
  virtual ~EplbPolicy() {};
  std::pair<torch::Tensor, std::vector<bool>> rebalance_experts(
      torch::Tensor expert_load);

 private:
  torch::Tensor old_expert_load_;
  int32_t device_experts_num_;
  int32_t device_num_;
  int32_t layer_num_;
  torch::Tensor expert_distribution_;
  torch::Tensor compute_balanced_pack(const torch::Tensor& expert_loads);
  std::pair<torch::Tensor, torch::Tensor> update_origin_weights(
      torch::Tensor expert_loads,
      int32_t redundancy_experts);
};
}  // namespace xllm
