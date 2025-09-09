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

#include "atb_base.h"

#include <glog/logging.h>

namespace xllm::hf {
static std::atomic<bool> g_executeOk(true);

ATBBase::ATBBase(const Context& context)
    : device_(context.get_tensor_options().device()),
      name_(""),
      parallel_args_(context.get_parallel_args()) {
  auto quant_args = context.get_quant_args();
  if (!quant_args.quantize_type().empty()) {
    quantize_type_ = quant_args.quantize_type();
  }

  if (!quant_args.torch_dtype().empty()) {
    torch_dtype_ = quant_args.torch_dtype();
  }

  dp_size_ = parallel_args_.dp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / dp_size_;
  dp_rank_ = parallel_args_.rank() / dp_local_tp_size_;
  CHECK_EQ(parallel_args_.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;

  runTaskFunc_ = std::bind(
      &ATBBase::run_task, this, std::placeholders::_1, std::placeholders::_2);
}

void ATBBase::print_atbtensor(const atb::Tensor& tensor, int i) {
  static std::map<aclDataType, at::ScalarType> dtypeMap = {
      {ACL_BOOL, at::ScalarType::Bool},
      {ACL_UINT8, at::ScalarType::Byte},
      {ACL_INT8, at::ScalarType::Char},
      {ACL_FLOAT16, at::ScalarType::Half},
      {ACL_FLOAT, at::ScalarType::Float},
      {ACL_INT32, at::ScalarType::Int},
      {ACL_INT64, at::ScalarType::Long},
      {ACL_BF16, at::ScalarType::BFloat16},
  };
  std::vector<int64_t> sizes;
  for (uint64_t i = 0; i < tensor.desc.shape.dimNum; i++) {
    sizes.push_back(tensor.desc.shape.dims[i]);
  }
  at::IntArrayRef sizeRef(sizes);
  at::ScalarType dtype;
  auto it = dtypeMap.find(tensor.desc.dtype);
  if (it != dtypeMap.end()) {
    dtype = it->second;
  } else {
    throw std::runtime_error("AtTensor2Tensor: not support dtype");
  }
  at::Tensor atTensor = torch::empty(sizeRef, at::TensorOptions().dtype(dtype));
  if (!atTensor.is_contiguous()) {
    atTensor = atTensor.contiguous();
  }
  aclrtMemcpy(atTensor.data_ptr(),
              tensor.dataSize,
              tensor.deviceData,
              tensor.dataSize,
              ACL_MEMCPY_DEVICE_TO_HOST);
}

torch::Dtype ATBBase::string_2_dtype(const std::string& dtype_str) {
  if (dtype_str == "float16") {
    return torch::kFloat16;
  } else if (dtype_str == "bfloat16") {
    return torch::kBFloat16;
  } else if (dtype_str == "float32") {
    return torch::kFloat32;
  } else if (dtype_str == "float64") {
    return torch::kFloat64;
  } else if (dtype_str == "int8") {
    return torch::kInt8;
  } else if (dtype_str == "int16") {
    return torch::kInt16;
  } else if (dtype_str == "int32") {
    return torch::kInt32;
  } else if (dtype_str == "int64") {
    return torch::kInt64;
  } else if (dtype_str == "uint8") {
    return torch::kUInt8;
  } else if (dtype_str == "bool") {
    return torch::kBool;
  }
  throw std::runtime_error("Unsupported dtype string");
}

atb::Tensor ATBBase::XTensor2Tensor(
    const std::shared_ptr<xllm::XTensor>& xtensor) {
  static std::map<at::ScalarType, aclDataType> dtypeMap = {
      {at::ScalarType::Bool, ACL_BOOL},
      {at::ScalarType::Byte, ACL_UINT8},
      {at::ScalarType::Char, ACL_INT8},
      {at::ScalarType::Half, ACL_FLOAT16},
      {at::ScalarType::Float, ACL_FLOAT},
      {at::ScalarType::Int, ACL_INT32},
      {at::ScalarType::Long, ACL_INT64},
      {at::ScalarType::BFloat16, ACL_BF16},
  };

  atb::Tensor tensor;
  // continuous kvcache only support ND format
  tensor.desc.format = ACL_FORMAT_ND;
  tensor.deviceData = xtensor->get_base_ptr();

  tensor.desc.shape.dimNum = 4;
  tensor.desc.shape.dims[0] = 0;
  tensor.desc.shape.dims[1] = 128;  // block_size
  tensor.desc.shape.dims[2] =
      xtensor->options().num_kv_heads();                       // num_kv_heads
  tensor.desc.shape.dims[3] = xtensor->options().head_size();  // head_size

  auto it = dtypeMap.find(xtensor->dtype());
  if (it != dtypeMap.end()) {
    tensor.desc.dtype = it->second;
  } else {
    throw std::runtime_error("XTensor2Tensor: not support dtype");
  }

  tensor.dataSize = 0;

  return tensor;
}

void ATBBase::correct_tensor_dtype(torch::Tensor& tensor,
                                   const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") && torch_dtype_ == "bfloat16") {
    return;
  }
  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string_2_dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

void ATBBase::set_weight(const StateDict& state_dict,
                         const std::string& tensor_name,
                         int weight_position) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
    }
  }
}

void ATBBase::set_weight(const StateDict& state_dict,
                         const std::string& tensor_name,
                         int weight_position,
                         int dim) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      bool use_non_sharded =
          (parallel_args_.world_size() <= 1) ||
          (tensor.size(dim) % parallel_args_.world_size() != 0);
      if (use_non_sharded) {
        if (tensor.size(dim) % parallel_args_.world_size() != 0 &&
            parallel_args_.world_size() > 1) {
          LOG(WARNING) << "Tensor '" << state_dict.prefix() << tensor_name
                       << "' dimension " << dim << " (size=" << tensor.size(dim)
                       << ") is not divisible by world_size="
                       << parallel_args_.world_size()
                       << ". Using non-sharded tensor loading.";
        }
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size())
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

void ATBBase::set_weight(const StateDict& state_dict,
                         const std::string& tensor_name,
                         int weight_position,
                         int dim,
                         int rank,
                         int world_size) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      if (world_size <= 1) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/rank,
                                    /*world_size=*/world_size)
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

void ATBBase::run_task(std::string taskName, std::function<int()> task) const {
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
}

atb::Status ATBBase::execute_node(atb_speed::Model::Node& node,
                                  atb::Context* context,
                                  AtbWorkspace& workspace,
                                  int nodeId,
                                  aclrtEvent* event,
                                  std::atomic<bool>* event_flag) {
  if (!g_executeOk) {
    std::stringstream ss;
    ss << "execute fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export "
          "ASDOPS_LOG_TO_STDOUT=1 to find the "
          "first error.  For more details, see the npu official document. "
       << std::endl;
    throw std::runtime_error(ss.str());
  }
  context_ = context;
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = workspace.GetWorkspaceBuffer(node.workspaceSize);
  }

  runTaskFunc_(name_ + std::to_string(nodeId), [=]() {
    return execute_plan(
        node, name_ + std::to_string(nodeId), event, event_flag);
  });

  return st;
}

atb::Status ATBBase::execute_plan(const atb_speed::Model::Node& node,
                                  std::string opName_,
                                  aclrtEvent* event,
                                  std::atomic<bool>* event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context_);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  if (st == 0 && event != nullptr) {
    aclrtStream stream = context_->GetExecuteStream();
    auto ret = aclrtRecordEvent(*event, stream);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Record event failed.";
      return st;
    }
    event_flag->store(true, std::memory_order_release);
  }
  return st;
}

}  // namespace xllm::hf
