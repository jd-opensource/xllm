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

#include "npu_base_layer.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"

namespace xllm {
namespace layer {

namespace {
inline size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}
}  // namespace

NpuBaseLayer::NpuBaseLayer(const ModelContext& context) : BaseLayer(context) {
  context_ = const_cast<atb::Context*>(context.get_atb_context());
  work_space_ = AtbWorkspace(device_);
}

NpuBaseLayer::~NpuBaseLayer() {
  release_host_storage();
  release_device_storage();
}

atb::Status NpuBaseLayer::execute_node(atb_speed::Model::Node& node,
                                       int node_id,
                                       aclrtEvent* event,
                                       std::atomic<bool>* event_flag) {
  // TODO（by zhangminchao1@jd.com): Stream management needs to be refactored
  // for better separation of concerns Current issues:
  // 1. ACLGraph capture requires execution on a non-default stream, so we
  // temporarily set the current stream
  // 2. After ACLGraph capture ends, the stream will be modified back to the
  // default stream
  // 3. In non-ACL graph capture mode, the context stream should be set to the
  // default stream
  // 4. The actual requirement is to separate decode node context from prefill
  // node context
  //
  // Note: The commented code below will cause runtime errors because:
  // - aclmdlRICaptureGetInfo() may fail when called at inappropriate times
  // - The capture status check logic is not robust enough for all scenarios
  // - Stream management conflicts: ATB context stream must be consistent with
  // libtorch_npu current stream.
  //   However, libtorch_npu current stream is set to default stream after
  //   capture ends, causing inconsistency between ATB context and the actual
  //   execution stream
  if (FLAGS_enable_acl_graph) {
    void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
    context_->SetExecuteStream(stream);
  }
  // if (FLAGS_enable_acl_graph && !graph_captured_) {
  //   void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
  //   aclmdlRICaptureStatus status;
  //   aclmdlRI modelRI;
  //   auto error = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
  //   if (error != ACL_SUCCESS) {
  //     LOG(ERROR) << "aclmdlRICaptureGetInfo failed, acl error code: " <<
  //     error;
  //   }
  //   if (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
  //     context_->SetExecuteStream(stream);
  //     graph_captured_ = true;
  //   }
  // }
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = work_space_.get_workspace_buffer(node.workspaceSize);
  }

  run_task_func_(name_ + std::to_string(node_id), [=]() {
    return execute_plan(
        node, name_ + std::to_string(node_id), event, event_flag);
  });

  return st;
}

atb::Status NpuBaseLayer::execute_plan(const atb_speed::Model::Node& node,
                                       const std::string& op_name,
                                       aclrtEvent* event,
                                       std::atomic<bool>* event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context_);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  if (st == 0 && event != nullptr) {
    aclrtStream stream = context_->GetExecuteStream();

    aclrtEvent* aclrt_event = reinterpret_cast<aclrtEvent*>(event);

    auto ret = aclrtRecordEvent(*aclrt_event, stream);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Record event failed.";
      return st;
    }

    event_flag->store(true, std::memory_order_release);
  }

  return st;
}

void NpuBaseLayer::run_task(std::string taskName,
                            std::function<int()> task) const {
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
}

void NpuBaseLayer::init_weight_slices(int weight_count) {
  weight_slices_.resize(weight_count);
  size_t offset = 0;
  for (size_t i = 0; i < weight_count; ++i) {
    weight_slices_[i] = {};
    const auto& tensor = at_host_weight_tensors_[i];
    if (!tensor.defined() || tensor.numel() < 1) {
      continue;
    }
    // offset = AlignUp(offset, kHostAlignment);
    weight_slices_[i].offset = offset;
    weight_slices_[i].bytes = tensor.nbytes();
    weight_slices_[i].sizes = tensor.sizes().vec();
    weight_slices_[i].dtype = tensor.scalar_type();
    offset += weight_slices_[i].bytes;
  }
  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  // storage_size_ = AlignUp(offset, max_alignment);
  LOG(INFO) << "NpuBaseLayer total weight size: " << offset;
  storage_size_ = offset;
}

void NpuBaseLayer::copy_weights_to_pinned_host() {
  CHECK_GT(storage_size_, 0) << "model size must be greater than 0.";
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(storage_size_, max_alignment);

  auto ret = aclrtMallocHost(&host_pinned_storage_, storage_size_);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate pinned host storage size=" << storage_size_;

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    auto host_tensor = at_host_weight_tensors_[i].to(torch::kCPU).contiguous();
    void* dst = static_cast<char*>(host_pinned_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    std::memcpy(dst, host_tensor.data_ptr(), slice.bytes);
    at_host_weight_tensors_[i] = at::Tensor();
  }

  ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate contiguous device storage size=" << storage_size_;
}

void NpuBaseLayer::copy_weights_to_device() {
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";
  auto ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate contiguous device storage size=" << storage_size_;

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* dst = static_cast<char*>(device_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    auto host_tensor = at_host_weight_tensors_[i].contiguous();
    auto err = aclrtMemcpy(dst,
                           slice.bytes,
                           host_tensor.data_ptr(),
                           slice.bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpy failed for tensor index " << i;
    at_host_weight_tensors_[i] = at::Tensor();
  }
}

torch::Tensor NpuBaseLayer::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const uintptr_t& dev_addr,
    int acl_format) {
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(dev_addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
  storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
  storage.set_data_ptr(std::move(c10_data_ptr));

  tensor.set_(storage, 0, dims);
  // cast npu format to nd
  tensor = at_npu::native::npu_format_cast(tensor, acl_format);

  return tensor;
}

void NpuBaseLayer::init_atb_tensors() {
  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* base = static_cast<char*>(device_storage_) +
                 static_cast<ptrdiff_t>(slice.offset);
    at_weight_tensors_[i] = convert_to_torch_tensor(
        slice.sizes, slice.dtype, reinterpret_cast<uintptr_t>(base));
  }

  c10_npu::NPUCachingAllocator::emptyCache();

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
}

void NpuBaseLayer::copy_weights_to_device_async() {
  CHECK_EQ(weight_slices_.size(), at_weight_tensors_.size())
      << "weight_slices_ size and at_weight_tensors_ size mismatch.";
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  void* dst = static_cast<char*>(device_storage_);
  void* src = static_cast<char*>(host_pinned_storage_);

  auto err = aclrtMemcpyAsync(dst,
                              storage_size_,
                              src,
                              storage_size_,
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              stream);
  CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpyAsync failed";
}

void NpuBaseLayer::release_device_storage() {
  if (device_storage_ == nullptr) {
    return;
  }
  auto ret = aclrtFree(device_storage_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to free contiguous layer storage, ret=" << ret;
  }
  device_storage_ = nullptr;
}

void NpuBaseLayer::release_host_storage() {
  if (host_pinned_storage_ == nullptr) {
    return;
  }
  auto ret = aclrtFreeHost(host_pinned_storage_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to free pinned host storage, ret=" << ret;
  }
  host_pinned_storage_ = nullptr;
}

atb::Tensor NpuBaseLayer::XTensor2Tensor(
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
    LOG(FATAL) << "XTensor2Tensor: not support dtype: " << xtensor->dtype();
  }

  tensor.dataSize = 0;

  return tensor;
}

}  // namespace layer
}  // namespace xllm