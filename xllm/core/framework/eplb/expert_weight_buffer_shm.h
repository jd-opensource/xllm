#pragma once

#include <fcntl.h>
#include <torch/torch.h>
#include <unistd.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "shared_memory_manager.h"

namespace xllm {

// Maximum number of tensors each expert-layer pair can store
constexpr int MAX_TENSORS_PER_LAYER = 16;
// Maximum number of layers per expert
constexpr int MAX_LAYERS_PER_EXPERT = 128;

// Shared memory header structure containing control information
struct SharedHeader {
  std::atomic<int64_t> initialized_layers;  // Number of initialized layers
  pthread_mutex_t allocation_mutex;  // Cross-process synchronization mutex
};

// Metadata structure for each stored tensor
struct TensorMeta {
  char tensor_name[256];  // Null-terminated tensor identifier
  int32_t rank;           // Number of dimensions (1D, 2D, etc.)
  int64_t shape[8];       // Dimensions of the tensor (max 8D)
  int32_t dtype;          // Data type (matches torch::Dtype)
  size_t data_offset;     // Byte offset in shared memory
  size_t actual_size;     // Unpadded data size in bytes
};

class ExpertBufferShm {
 public:
  ExpertBufferShm(int32_t expert_id, int32_t max_layers, int64_t total_size);

  virtual ~ExpertBufferShm();

  void add_tensor(int32_t layer_id,
                  const std::string& tensor_name,
                  const torch::Tensor& tensor);

  /**
   * @brief Retrieve a tensor from expert's layer memory region
   *
   * @param layer_id Source layer identifier
   * @param tensor_name Name of the tensor to retrieve
   * @return torch::Tensor A copy of the requested tensor
   * @throws std::runtime_error if tensor not found
   */
  torch::Tensor get_tensor(int32_t layer_id, const std::string& tensor_name);

 private:
  // Initializes shared memory when creating new region
  void initialize_as_creator();

  // Verifies and recovers shared memory state
  void verify_and_recover();

  // Calculates base offset for a layer's data region
  size_t get_layer_offset(int32_t layer_id) const;

  std::mutex local_mutex_;                    // Thread synchronization
  std::unique_ptr<SharedMemoryManager> shm_;  // Shared memory manager
  SharedHeader* header_ = nullptr;            // Pointer to shared header
  TensorMeta* tensor_metas_ = nullptr;        // Array of all layers' metadata
  char* data_base_ = nullptr;                 // Base pointer to data region

  const int32_t expert_id_;               // Expert identifier
  const int32_t max_layers_;              // Maximum supported layers
  const int64_t layer_data_region_size_;  // Bytes allocated per layer
};

}  // namespace xllm