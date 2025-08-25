#include "expert_buffer_manager.h"

namespace xllm {

ExpertBufferManager::ExpertBufferManager(int32_t num_experts,
                                         int32_t num_layers,
                                         int64_t shm_size_per_expert)
    : num_experts_(num_experts),
      num_layers_(num_layers),
      shm_size_per_expert_(shm_size_per_expert) {
  expert_buffers_.reserve(num_experts);
  for (int32_t i = 0; i < num_experts; ++i) {
    expert_buffers_.emplace_back(
        std::make_unique<ExpertBufferShm>(i, num_layers, shm_size_per_expert));
  }
}

void ExpertBufferManager::add_tensor(int32_t expert_id,
                                     int32_t layer_id,
                                     const std::string& tensor_name,
                                     const torch::Tensor& tensor) {
  if (expert_id < 0 || expert_id >= num_experts_) {
    throw std::runtime_error("Invalid expert ID: " + std::to_string(expert_id));
  }
  expert_buffers_[expert_id]->add_tensor(layer_id, tensor_name, tensor);
}

torch::Tensor ExpertBufferManager::get_tensor(int32_t expert_id,
                                              int32_t layer_id,
                                              const std::string& tensor_name) {
  if (expert_id < 0 || expert_id >= num_experts_) {
    throw std::runtime_error("Invalid expert ID: " + std::to_string(expert_id));
  }
  return expert_buffers_[expert_id]->get_tensor(layer_id, tensor_name);
}

}  // namespace xllm