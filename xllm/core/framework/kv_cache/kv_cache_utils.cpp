/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_cache_utils.h"

#include <glog/logging.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "util/utils.h"
#if defined(USE_MLU)
#include "platform/mlu/mlu_tensor_alloc.h"
#endif
#if defined(USE_NPU)
#include "acl/acl_rt.h"

extern "C" aclError aclrtHostRegister(void* ptr,
                                      uint64_t size,
                                      aclrtHostRegisterType type,
                                      void** devPtr);
extern "C" aclError aclrtHostUnregister(void* ptr);
#endif

namespace xllm {
namespace {

size_t get_tensor_nbytes(const std::vector<int64_t>& dims,
                         torch::ScalarType dtype) {
  size_t count = 1;
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "tensor dim must be non-negative";
    const size_t dim_size = static_cast<size_t>(dim);
    if (dim_size > 0) {
      CHECK_LE(count, std::numeric_limits<size_t>::max() / dim_size)
          << "tensor element count overflow";
    }
    count *= dim_size;
  }
  const size_t elem_size = static_cast<size_t>(torch::elementSize(dtype));
  CHECK_GT(elem_size, static_cast<size_t>(0)) << "tensor dtype size is zero";
  CHECK_LE(count, std::numeric_limits<size_t>::max() / elem_size)
      << "tensor byte size overflow";
  return count * elem_size;
}

size_t get_host_page_size() {
  const long page_size = sysconf(_SC_PAGESIZE);
  CHECK_GT(page_size, 0) << "sysconf(_SC_PAGESIZE) failed.";
  return static_cast<size_t>(page_size);
}

size_t get_page_aligned_bytes(size_t bytes) {
  CHECK_GT(bytes, static_cast<size_t>(0))
      << "page-aligned host memory bytes must be positive.";
  const size_t page_size = get_host_page_size();
  CHECK_LE(bytes, std::numeric_limits<size_t>::max() - (page_size - 1))
      << "host memory byte size overflow during page alignment.";
  return ((bytes + page_size - 1) / page_size) * page_size;
}

#if defined(USE_NPU)
void free_acl_tensor(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  const auto acl_ret = aclrtFree(ptr);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtFree failed, ret=" << std::hex << acl_ret << ", ptr=" << ptr;
}

torch::Tensor alloc_npu_huge_page_tensor(const std::vector<int64_t>& dims,
                                         torch::ScalarType dtype,
                                         aclFormat format) {
  void* buffer = nullptr;
  const size_t nbytes = get_tensor_nbytes(dims, dtype);
  auto acl_ret = aclrtMalloc(&buffer, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtMalloc KV cache failed, ret=" << std::hex << acl_ret
      << ", nbytes=" << nbytes;

  constexpr c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  auto tensor = torch::empty(
      {0}, torch::TensorOptions().dtype(dtype).device(device_type));
  torch::DataPtr data_ptr(buffer, buffer, free_acl_tensor, tensor.device());

  auto* storage_create = c10::GetStorageImplCreate(device_type);
  auto* allocator = c10::GetAllocator(device_type);
  torch::Storage storage = storage_create(c10::StorageImpl::use_byte_size_t(),
                                          c10::SymInt(nbytes),
                                          std::move(data_ptr),
                                          allocator,
                                          true);

  tensor.set_(storage, 0, dims);
  auto* tensor_storage = static_cast<torch_npu::NPUStorageImpl*>(
      tensor.storage().unsafeGetStorageImpl());
  tensor_storage->npu_desc_.npu_format_ = format;
  return tensor;
}
#endif

}  // namespace

HostPageAlignedRegion::HostPageAlignedRegion(size_t bytes) {
  total_bytes = get_page_aligned_bytes(bytes);
  void* ptr = mmap(nullptr,
                   total_bytes,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1,
                   0);
  CHECK(ptr != MAP_FAILED)
      << "Failed to allocate page-aligned host memory region.";

  if (mlock(ptr, total_bytes) != 0) {
    const int err = errno;
    munmap(ptr, total_bytes);
    LOG(FATAL) << "Failed to lock page-aligned host memory region, errno="
               << err << " (" << strerror(err) << ")";
  }

#if defined(USE_NPU)
  void* mapped_ptr = nullptr;
  const aclError ret = aclrtHostRegister(
      ptr, total_bytes, ACL_HOST_REGISTER_MAPPED, &mapped_ptr);
  if (ret != ACL_SUCCESS) {
    munlock(ptr, total_bytes);
    munmap(ptr, total_bytes);
  }
  CHECK_EQ(ret, ACL_SUCCESS) << "aclrtHostRegister fail: " << ret;
#endif

  base_ptr = ptr;
}

HostPageAlignedRegion::HostPageAlignedRegion(
    HostPageAlignedRegion&& other) noexcept
    : base_ptr(other.base_ptr), total_bytes(other.total_bytes) {
  other.base_ptr = nullptr;
  other.total_bytes = 0;
}

namespace {

void release_host_page_aligned_region(void*& base_ptr, size_t& total_bytes) {
  if (base_ptr == nullptr || total_bytes == 0) {
    base_ptr = nullptr;
    total_bytes = 0;
    return;
  }
#if defined(USE_NPU)
  aclrtHostUnregister(base_ptr);
#endif
  munlock(base_ptr, total_bytes);
  munmap(base_ptr, total_bytes);
  base_ptr = nullptr;
  total_bytes = 0;
}

}  // namespace

HostPageAlignedRegion& HostPageAlignedRegion::operator=(
    HostPageAlignedRegion&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release_host_page_aligned_region(base_ptr, total_bytes);
  base_ptr = other.base_ptr;
  total_bytes = other.total_bytes;
  other.base_ptr = nullptr;
  other.total_bytes = 0;
  return *this;
}

HostPageAlignedRegion::~HostPageAlignedRegion() {
  release_host_page_aligned_region(base_ptr, total_bytes);
}

bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval) {
  if (full_attention_interval <= 1) {
    return false;
  }
  return (layer_idx + 1) % full_attention_interval != 0;
}

bool use_npu_nz_kv_cache_layout(const std::string& model_type) {
  return (model_type == "deepseek_v3" || model_type == "deepseek_v3_mtp") &&
         ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
}

int64_t scale_host_block_count(int64_t block_count, double host_blocks_factor) {
  CHECK_GT(block_count, 0) << "block_count must be positive.";
  const double factor = std::max(host_blocks_factor, 1.0);
  return std::max<int64_t>(block_count,
                           static_cast<int64_t>(block_count * factor));
}

std::vector<int64_t> build_host_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor) {
  CHECK(!base_shape.empty()) << "base_shape must not be empty.";
  std::vector<int64_t> host_shape = base_shape;
  host_shape[0] = scale_host_block_count(host_shape[0], host_blocks_factor);
  return host_shape;
}

std::vector<int64_t> build_host_group_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor,
    int64_t layer_count) {
  CHECK_GT(layer_count, 0) << "layer_count must be positive.";
  std::vector<int64_t> host_shape =
      build_host_tensor_shape(base_shape, host_blocks_factor);
  host_shape.insert(host_shape.begin() + 1, layer_count);
  return host_shape;
}

int64_t resolve_host_group_layer_count(PrefixCacheGroup group,
                                       const KVCacheCreateOptions& options) {
  CHECK_GT(options.num_layers(), 0) << "num_layers must be positive.";
  int64_t layer_count = 0;
  switch (group) {
    case PrefixCacheGroup::C1:
      if (!options.enable_linear_attention()) {
        return options.num_layers();
      }
      for (int64_t layer_idx = 0; layer_idx < options.num_layers();
           ++layer_idx) {
        if (!is_linear_attention_layer(layer_idx,
                                       options.full_attention_interval())) {
          ++layer_count;
        }
      }
      return layer_count;
    case PrefixCacheGroup::SINGLE:
      CHECK(options.enable_linear_attention())
          << "SINGLE host group requires linear attention to be enabled.";
      for (int64_t layer_idx = 0; layer_idx < options.num_layers();
           ++layer_idx) {
        if (is_linear_attention_layer(layer_idx,
                                      options.full_attention_interval())) {
          ++layer_count;
        }
      }
      return layer_count;
    default:
      LOG(FATAL) << "Unsupported non-DSV4 host prefix cache group: "
                 << group.to_string();
  }
}

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  KVCacheTensors tensors;
#if defined(USE_MLU)
  if (create_options.enable_raw_device_allocator()) {
    tensors.key_cache = mlu::alloc_zero_tensor(kv_cache_shape.key_cache_shape(),
                                               create_options.dtype(),
                                               create_options.device());
    if (kv_cache_shape.has_value_cache_shape()) {
      tensors.value_cache =
          mlu::alloc_zero_tensor(kv_cache_shape.value_cache_shape(),
                                 create_options.dtype(),
                                 create_options.device());
    }
  } else {
    tensors.key_cache = torch::zeros(
        kv_cache_shape.key_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
    if (!kv_cache_shape.value_cache_shape().empty()) {
      tensors.value_cache = torch::zeros(
          kv_cache_shape.value_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device()));
    }
  }
#elif defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.key_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.key_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
    tensors.value_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.value_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
  } else {
    tensors.key_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.key_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
    tensors.value_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.value_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
  }
#else
  tensors.key_cache = torch::zeros(
      kv_cache_shape.key_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));

  // deepseek_v3 model has no value cache on mlu device
  if (!kv_cache_shape.value_cache_shape().empty()) {
    tensors.value_cache = torch::zeros(
        kv_cache_shape.value_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
  }
#endif
  return tensors;
}

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_index_cache_shape())
      << "index_cache_shape must be initialized.";
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

#if defined(USE_MLU)
  if (create_options.enable_raw_device_allocator()) {
    tensors.index_cache =
        mlu::alloc_zero_tensor(kv_cache_shape.index_cache_shape(),
                               create_options.dtype(),
                               create_options.device());
  } else {
    tensors.index_cache = torch::zeros(
        kv_cache_shape.index_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
  }
#elif defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.index_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.index_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
  } else {
    tensors.index_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.index_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
  }
#else
  tensors.index_cache = torch::zeros(
      kv_cache_shape.index_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
#endif
  return tensors;
}

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
#if !defined(USE_MLU)
  CHECK(!create_options.enable_kv_cache_quant())
      << "KV cache quantization is only supported on MLU backend.";
#endif

  QuantizedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  std::vector<int64_t> key_scale_shape(key_cache_shape.begin(),
                                       key_cache_shape.end() - 1);

  // float32 scale tensor for quantized KV cache (int8)
  tensors.key_cache_scale = torch::zeros(
      key_scale_shape,
      torch::dtype(torch::kFloat32).device(create_options.device()));
  if (!kv_cache_shape.value_cache_shape().empty()) {
    const std::vector<int64_t>& value_cache_shape =
        kv_cache_shape.value_cache_shape();
    std::vector<int64_t> value_scale_shape(value_cache_shape.begin(),
                                           value_cache_shape.end() - 1);
    tensors.value_cache_scale = torch::zeros(
        value_scale_shape,
        torch::dtype(torch::kFloat32).device(create_options.device()));
  }

  return tensors;
}

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_conv_cache_shape())
      << "conv_cache_shape must be initialized.";
  CHECK(kv_cache_shape.has_ssm_cache_shape())
      << "ssm_cache_shape must be initialized.";
  LinearAttentionKVCacheTensors tensors;

#if defined(USE_NPU)
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.conv_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.conv_cache_shape(),
                                   create_options.dtype(),
                                   ACL_FORMAT_ND);
    tensors.conv_cache.zero_();
    tensors.ssm_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.ssm_cache_shape(),
                                   create_options.ssm_dtype(),
                                   ACL_FORMAT_ND);
    tensors.ssm_cache.zero_();
  } else {
    tensors.conv_cache = at_npu::native::npu_format_cast(
        torch::zeros(kv_cache_shape.conv_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        ACL_FORMAT_ND);
    tensors.ssm_cache = at_npu::native::npu_format_cast(
        torch::zeros(kv_cache_shape.ssm_cache_shape(),
                     torch::dtype(create_options.ssm_dtype())
                         .device(create_options.device())),
        ACL_FORMAT_ND);
  }
#else
  tensors.conv_cache = torch::zeros(
      kv_cache_shape.conv_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
  tensors.ssm_cache = torch::zeros(
      kv_cache_shape.ssm_cache_shape(),
      torch::dtype(create_options.ssm_dtype()).device(create_options.device()));
#endif

  return tensors;
}

void create_host_page_aligned_tensor(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     torch::Tensor* tensor,
                                     HostPageAlignedRegion* region) {
  CHECK(tensor != nullptr) << "tensor must not be null.";
  CHECK(region != nullptr) << "region must not be null.";
  const size_t tensor_bytes = get_tensor_nbytes(dims, dtype);
  CHECK_GT(tensor_bytes, static_cast<size_t>(0))
      << "host cache tensor bytes must be positive.";

  *region = HostPageAlignedRegion(tensor_bytes);
  std::memset(region->base_ptr, 0, region->total_bytes);
  *tensor = torch::from_blob(
      region->base_ptr,
      dims,
      torch::TensorOptions().dtype(dtype).device(torch::Device(torch::kCPU)));
  CHECK(tensor->is_contiguous()) << "host cache tensor must be contiguous.";
}

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type) {
  return use_npu_nz_kv_cache_layout(model_type) ? ACL_FORMAT_FRACTAL_NZ
                                                : ACL_FORMAT_ND;
}
#endif

}  // namespace xllm
