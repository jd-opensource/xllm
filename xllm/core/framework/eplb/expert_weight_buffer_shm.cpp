#include "expert_weight_buffer_shm.h"

#include <fmt/format.h>

#include <cstring>
#include <mutex>
#include <thread>
namespace xllm {

ExpertBufferShm::ExpertBufferShm(int32_t expert_id,
                                 int32_t max_layers,
                                 int64_t total_size)
    : expert_id_(expert_id),
      max_layers_(max_layers),
      layer_data_region_size_(total_size / max_layers) {
  // Memory alignment calculation (64-byte alignment for performance)
  constexpr size_t kAlignment = 64;

  // Calculate aligned header size (header + padding)
  size_t header_size =
      ((sizeof(SharedHeader) + kAlignment - 1) / kAlignment) * kAlignment;

  // Calculate aligned metadata region size (all experts' metadata + padding)
  size_t meta_size = ((max_layers * MAX_TENSORS_PER_LAYER * sizeof(TensorMeta) +
                       kAlignment - 1) /
                      kAlignment) *
                     kAlignment;

  bool is_creator;
  std::string shm_name = "xllm_expert_" + std::to_string(expert_id_);

  // Create/attach shared memory segment with calculated size
  shm_ = std::make_unique<SharedMemoryManager>(
      shm_name, header_size + meta_size + total_size, is_creator);

  // Memory region pointers setup:
  header_ = static_cast<SharedHeader*>(shm_->base_address());
  tensor_metas_ = reinterpret_cast<TensorMeta*>(
      static_cast<char*>(shm_->base_address()) + header_size);
  data_base_ =
      static_cast<char*>(shm_->base_address()) + header_size + meta_size;

  if (is_creator) {
    initialize_as_creator();
  }
  verify_and_recover();
}

ExpertBufferShm::~ExpertBufferShm() {
  std::lock_guard<std::mutex> lock(local_mutex_);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  header_ = nullptr;
  tensor_metas_ = nullptr;
  data_base_ = nullptr;
}

void ExpertBufferShm::initialize_as_creator() {
  header_->initialized_layers.store(0, std::memory_order_release);

  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);

  if (pthread_mutex_init(&header_->allocation_mutex, &attr) != 0) {
    pthread_mutexattr_destroy(&attr);
    throw std::runtime_error("Mutex initialization failed");
  }
  pthread_mutexattr_destroy(&attr);

  memset(tensor_metas_,
         0,
         max_layers_ * MAX_TENSORS_PER_LAYER * sizeof(TensorMeta));
}

void ExpertBufferShm::verify_and_recover() {
  int rc = pthread_mutex_lock(&header_->allocation_mutex);
  if (rc == EOWNERDEAD) {
    pthread_mutex_consistent(&header_->allocation_mutex);
    LOG(WARNING) << "Recovered from orphaned mutex for expert " << expert_id_;
  } else if (rc != 0) {
    throw std::runtime_error("Failed to acquire mutex");
  }
  pthread_mutex_unlock(&header_->allocation_mutex);
}

size_t ExpertBufferShm::get_layer_offset(int32_t layer_id) const {
  if (layer_id < 0 || layer_id >= max_layers_) {
    throw std::runtime_error("Invalid layer ID: " + std::to_string(layer_id) +
                             " for expert " + std::to_string(expert_id_));
  }
  return layer_id * layer_data_region_size_;
}

void ExpertBufferShm::add_tensor(int32_t layer_id,
                                 const std::string& tensor_name,
                                 const torch::Tensor& tensor) {
  if (layer_id < 0 || layer_id >= max_layers_) {
    throw std::runtime_error("Invalid layer ID: " + std::to_string(layer_id) +
                             " for expert " + std::to_string(expert_id_));
  }
  if (tensor_name.empty()) {
    throw std::runtime_error("Tensor name cannot be empty");
  }
  if (!tensor.defined() || !tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be defined and contiguous");
  }
  if (tensor.device().type() != torch::kCPU) {
    throw std::runtime_error("Only CPU tensors can be stored in shared memory");
  }

  std::lock_guard<std::mutex> lock(local_mutex_);

  // Get this expert's metadata block
  TensorMeta* layer_metas = &tensor_metas_[layer_id * MAX_TENSORS_PER_LAYER];

  // Find available slot and check for duplicates
  int available_slot = -1;
  for (int i = 0; i < MAX_TENSORS_PER_LAYER; ++i) {
    TensorMeta& meta = layer_metas[i];
    if (meta.tensor_name[0] == '\0') {
      if (available_slot == -1) available_slot = i;
    } else if (strcmp(meta.tensor_name, tensor_name.c_str()) == 0) {
      throw std::runtime_error(
          "Tensor '" + tensor_name + "' already exists for expert " +
          std::to_string(expert_id_) + " layer " + std::to_string(layer_id));
    }
  }

  if (available_slot == -1) {
    throw std::runtime_error("No available slots for expert " +
                             std::to_string(expert_id_) + " layer " +
                             std::to_string(layer_id));
  }

  // Prepare the tensor metadata
  TensorMeta& meta = layer_metas[available_slot];
  strncpy(meta.tensor_name, tensor_name.c_str(), sizeof(meta.tensor_name) - 1);
  meta.tensor_name[sizeof(meta.tensor_name) - 1] = '\0';

  meta.rank = tensor.dim();
  for (int i = 0; i < meta.rank; ++i) {
    meta.shape[i] = tensor.size(i);
  }
  meta.dtype = static_cast<int32_t>(tensor.scalar_type());

  constexpr size_t alignment = 64;
  size_t raw_size = tensor.nbytes();
  size_t aligned_size = (raw_size + alignment - 1) & ~(alignment - 1);

  // Calculate offset by summing sizes of previous tensors in this expert
  size_t layer_data_offset = 0;
  for (int i = 0; i < MAX_TENSORS_PER_LAYER; ++i) {
    if (&layer_metas[i] == &meta) break;
    layer_data_offset += layer_metas[i].actual_size;
  }

  if (layer_data_offset + aligned_size > layer_data_region_size_) {
    throw std::runtime_error(
        "Insufficient space in expert " + std::to_string(expert_id_) +
        " layer " + std::to_string(layer_id) + " (needs " +
        std::to_string(aligned_size) + " bytes, has " +
        std::to_string(layer_data_region_size_ - layer_data_offset) +
        " remaining)");
  }

  // Set final storage location
  meta.data_offset = get_layer_offset(layer_id) + layer_data_offset;
  meta.actual_size = raw_size;

  // Copy tensor data to shared memory
  void* dest = data_base_ + meta.data_offset;
  memcpy(dest, tensor.data_ptr(), raw_size);

  // Zero-fill any alignment padding
  if (aligned_size > raw_size) {
    memset(static_cast<char*>(dest) + raw_size, 0, aligned_size - raw_size);
  }
}

torch::Tensor ExpertBufferShm::get_tensor(int32_t layer_id,
                                          const std::string& tensor_name) {
  if (layer_id < 0 || layer_id >= max_layers_) {
    throw std::runtime_error(
        fmt::format("Invalid layer ID {} for expert {}", layer_id, expert_id_));
  }

  // Validate expert ID
  std::lock_guard<std::mutex> lock(local_mutex_);

  // Get this expert's metadata block
  TensorMeta* layer_metas = &tensor_metas_[layer_id * MAX_TENSORS_PER_LAYER];

  // Search for the requested tensor
  for (int i = 0; i < MAX_TENSORS_PER_LAYER; ++i) {
    TensorMeta& meta = layer_metas[i];

    // Skip empty slots
    if (meta.tensor_name[0] == '\0') {
      continue;
    }

    // Check for name match
    if (strcmp(meta.tensor_name, tensor_name.c_str()) == 0) {
      // Validate metadata
      if (meta.data_offset < 0 || meta.actual_size == 0 ||
          meta.data_offset + meta.actual_size > shm_->size()) {
        throw std::runtime_error(fmt::format(
            "Corrupted tensor metadata for {} in expert {} layer {}",
            tensor_name,
            expert_id_,
            layer_id));
      }

      // Create tensor options from stored type
      auto options = torch::TensorOptions()
                         .dtype(static_cast<torch::ScalarType>(meta.dtype))
                         .device(torch::kCPU)
                         .requires_grad(false);

      // Convert shape array to vector
      std::vector<int64_t> shape(meta.shape, meta.shape + meta.rank);

      // Create tensor from shared memory
      void* src = data_base_ + meta.data_offset;
      torch::Tensor result = torch::from_blob(src, shape, options);

      return result;
    }
  }

  throw std::runtime_error(
      fmt::format("Tensor {} not found in expert {} layer {}",
                  tensor_name,
                  expert_id_,
                  layer_id));
}

}  // namespace xllm