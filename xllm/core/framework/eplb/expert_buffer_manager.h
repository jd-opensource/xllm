#pragma once

#include <torch/torch.h>

#include <vector>

#include "expert_weight_buffer_shm.h"

namespace xllm {

class ExpertBufferManager {
 public:
  ExpertBufferManager(int32_t num_experts,
                      int32_t num_layers,
                      int64_t shm_size_per_expert);

  void add_tensor(int32_t expert_id,
                  int32_t layer_id,
                  const std::string& tensor_name,
                  const torch::Tensor& tensor);

  torch::Tensor get_tensor(int32_t expert_id,
                           int32_t layer_id,
                           const std::string& tensor_name);

 private:
  std::vector<std::unique_ptr<ExpertBufferShm>> expert_buffers_;
  const int32_t num_experts_;
  const int32_t num_layers_;
  const int64_t shm_size_per_expert_;
};

}  // namespace xllm
