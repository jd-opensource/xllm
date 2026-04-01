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

#include "base_loader.h"

namespace xllm {
namespace layer {

BaseLoader::BaseLoader(uint64_t weight_count, const ModelContext& context)
    : weight_count_(weight_count),
      parallel_args_(context.get_parallel_args()),
      device_(context.get_tensor_options().device()) {
  auto quant_args = context.get_quant_args();
  if (!quant_args.quantize_type().empty()) {
    quantize_type_ = quant_args.quantize_type();
  }

  if (!quant_args.torch_dtype().empty()) {
    torch_dtype_ = quant_args.torch_dtype();
  }

  dp_size_ = parallel_args_.dp_size();
  cp_size_ = parallel_args_.cp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / (dp_size_ * cp_size_);
  dp_rank_ = parallel_args_.rank() / dp_local_tp_size_ * cp_size_;
  CHECK_EQ(parallel_args_.world_size(),
           dp_size_ * dp_local_tp_size_ * cp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;

  at_weight_tensors_.resize(weight_count_);
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (parallel_args_.world_size() <= 1) {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  } else {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = state_dict.get_sharded_tensor(
            tensor_name,
            /*dim=*/dim,
            /*rank=*/parallel_args_.rank(),
            /*world_size=*/parallel_args_.world_size());
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim,
                            int rank,
                            int world_size,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (world_size <= 1) {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  } else {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor =
            state_dict.get_sharded_tensor(tensor_name,
                                          /*dim=*/dim,
                                          /*rank=*/rank,
                                          /*world_size=*/world_size);
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  }
}

void BaseLoader::correct_tensor_dtype(torch::Tensor& tensor,
                                      const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") &&
      (torch_dtype_.compare("bfloat16") == 0)) {
    return;
  }

  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string2dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

torch::Dtype BaseLoader::string2dtype(const std::string& dtype_str) {
  if (dtype_str.compare("float16") == 0) {
    return torch::kFloat16;
  } else if (dtype_str.compare("bfloat16") == 0) {
    return torch::kBFloat16;
  } else if (dtype_str.compare("float32") == 0) {
    return torch::kFloat32;
  } else if (dtype_str.compare("float64") == 0) {
    return torch::kFloat64;
  } else if (dtype_str.compare("int8") == 0) {
    return torch::kInt8;
  } else if (dtype_str.compare("int16") == 0) {
    return torch::kInt16;
  } else if (dtype_str.compare("int32") == 0) {
    return torch::kInt32;
  } else if (dtype_str.compare("int64") == 0) {
    return torch::kInt64;
  } else if (dtype_str.compare("uint8") == 0) {
    return torch::kUInt8;
  } else if (dtype_str.compare("bool") == 0) {
    return torch::kBool;
  }

  LOG(FATAL) << "Unsupported dtype string: " << dtype_str;
}

at::Tensor BaseLoader::pad_vocab_tensor(const at::Tensor& tensor,
                                        int64_t padded_vocab_size) const {
  if (tensor.size(0) >= padded_vocab_size) {
    return tensor;
  }
  at::Tensor padded_tensor =
      torch::zeros({padded_vocab_size, tensor.size(1)}, tensor.options());
  padded_tensor.slice(0, 0, tensor.size(0)) = tensor;
  return padded_tensor;
}

at::Tensor BaseLoader::shard_padded_tensor(const at::Tensor& padded_tensor,
                                           int dim,
                                           int rank,
                                           int world_size) const {
  if (world_size <= 1) {
    return padded_tensor;
  }
  auto chunks = padded_tensor.chunk(world_size, dim);
  return chunks[rank];
}

void BaseLoader::set_weight_with_padding(const StateDict& state_dict,
                                         const std::string& tensor_name,
                                         int weight_position,
                                         int dim,
                                         int64_t padded_vocab_size,
                                         bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      if (padded_vocab_size > tensor.size(0)) {
        mutable_tensor = pad_vocab_tensor(tensor, padded_vocab_size);
      }
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

void BaseLoader::set_weight_with_padding(const StateDict& state_dict,
                                         const std::string& tensor_name,
                                         int weight_position,
                                         int dim,
                                         int rank,
                                         int world_size,
                                         int64_t padded_vocab_size,
                                         bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (world_size <= 1) {
    set_weight_with_padding(state_dict,
                            tensor_name,
                            weight_position,
                            dim,
                            padded_vocab_size,
                            to_host);
    return;
  }
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      if (padded_vocab_size > tensor.size(0)) {
        // Memory-optimized path for vocabulary dimension sharding
        if (dim == 0) {
          int64_t shard_size = padded_vocab_size / world_size;
          int64_t start_idx = rank * shard_size;
          int64_t end_idx = (rank + 1) * shard_size;
          if (start_idx >= tensor.size(0)) {
            mutable_tensor =
                torch::zeros({shard_size, tensor.size(1)}, tensor.options());
          } else {
            auto valid_part =
                tensor.slice(0, start_idx, std::min(end_idx, tensor.size(0)));
            if (valid_part.size(0) < shard_size) {
              mutable_tensor =
                  torch::zeros({shard_size, tensor.size(1)}, tensor.options());
              mutable_tensor.slice(0, 0, valid_part.size(0)).copy_(valid_part);
            } else {
              mutable_tensor = valid_part.clone();
            }
          }
        } else {
          // Non-vocabulary dimension: use original approach
          mutable_tensor = pad_vocab_tensor(tensor, padded_vocab_size);
          mutable_tensor =
              shard_padded_tensor(mutable_tensor, dim, rank, world_size);
        }
      } else {
        mutable_tensor =
            state_dict.get_sharded_tensor(tensor_name, dim, rank, world_size);
      }
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

int64_t BaseLoader::get_padded_vocab_size(const ModelContext& context) const {
  int64_t vocab_size = context.get_model_args().vocab_size();
  int32_t local_tp_size = dp_local_tp_size_;
  if (vocab_size > 0 && local_tp_size > 1 && vocab_size % local_tp_size != 0) {
    return ((vocab_size + local_tp_size - 1) / local_tp_size) * local_tp_size;
  }
  return vocab_size;
}

}  // namespace layer
}  // namespace xllm
