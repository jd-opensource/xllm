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
  // Initialize policy engine parameters:
  // - device_experts_num: Experts per device (including redundancy)
  // - device_num: Total parallel devices
  // - layer_num: Model layers to manage
  EplbPolicy(int32_t device_experts_num, int32_t device_num, int32_t layer_num);

  virtual ~EplbPolicy() {};

  // Recalculate expert distribution based on latest workload
  // Input: expert_load - Workload tensor [total_experts]
  // Returns: <expert_distribution, update_flags> pair
  //          expert_distribution: [layers x devices x local_experts]
  //          update_flags: Boolean array marking layers needing update
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
