/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <ATen/DynamicLibrary.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "musa_tvmffi_stream.h"

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#endif

namespace ffi = tvm::ffi;

namespace xllm::kernel::cuda {

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

enum class ActivationType : int8_t {
  GELU = 0,
  RELU = 1,
  SILU = 2,
  SWIGLU = 3,
  GEGLU = 4,
  SWIGLU_BIAS = 5,
  RELU2 = 6,
  IDENTITY = 7,
  INVALID_TYPE = 8
};

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device);

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define DISPATCH_CASE_HALF_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
// NOLINTEND(cppcoreguidelines-macro-usage)

bool should_use_tensor_core(torch::ScalarType kv_cache_dtype,
                            int64_t num_attention_heads,
                            int64_t num_kv_heads);

bool support_pdl();

std::string path_to_uri_so_lib(const std::string& uri);

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask);

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction);

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap);

std::tuple<torch::Tensor, double> split_scale_param(const torch::Tensor& scale);

DLDataType to_dl_data_type(torch::ScalarType scalar_type);

// below are tvm-ffi related functions
ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor);

ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional);

ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors);

ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional);

ffi::Module get_module(const std::string& uri);

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name);

// =================================================================
// TVM-FFI allocation record / replay (Mate FFI on CUDA / MUSA Graph)
// =================================================================
// The Mate FFI decode / prefill `.so` libraries allocate per-call scratch
// tensors via the TVM-FFI `DLPackManagedTensorAllocator` hook. xLLM installs
// `torch_dlpack_managed_tensor_allocator` (see utils.cpp) as that hook, which
// normally forwards each request to `torch::empty(...)`.
//
// Under MUSA / CUDA stream capture, `torch::empty` eventually calls the
// caching allocator's `alloc_block`, which the driver rejects with
// "operation not permitted when stream is capturing" whenever it needs to
// grow a new physical mapping. Workarounds based on PyTorch's VMM MemPool
// require deep platform integration that is not yet implemented for
// `XLLM_TORCH_MUSA` builds (see cuda_graph_executor_impl.cpp line ~1248).
//
// The record/replay protocol below sidesteps that by capturing the exact
// sequence of FFI tensor handles during an extra eager-mode warmup, then
// serving the captured replay phase from the recording (no `torch::empty`
// is called inside the captured region).
//
// Diagnostic trace (eager mode, Qwen3.5-27B, max_tokens=8) showed:
//   * 1139 total FFI allocations across the request
//   * Per `batch_decode` call (decode path): 8 deterministic allocations
//     totalling ~24 KB (largest = `[1,24,1,1,256]` Float = 24576 B).
//   * Shapes scale linearly with the q_len / batch dim, so the recording
//     is also valid across replays of the same bucket.
//   * Order is identical across all 128 decode calls observed.
//
// Lifetime contract (critical -- see CudaGraph member `recorded_ffi_allocs_`):
//   * The captured graph holds device pointers into the recorded tensors.
//   * The caller MUST keep the vector alive until the captured graph is
//     destroyed; we attach it to `CudaGraph` so its destruction precedes
//     the graph handle's release.
//
// Thread-locality: state is `thread_local` -- the worker thread that runs
// `CudaGraph::capture()` is also the one that invokes `model->forward()`
// (and therefore the FFI hook), so the same thread sees both record and
// replay phases. Concurrent capture on the same executor instance is
// disallowed (see CudaGraphExecutorImpl::graph_pool_ comment).

enum class FfiAllocMode { kPassthrough, kRecord, kReplay };

// Switch the FFI allocator hook to recording mode. Subsequent allocations
// are still backed by `torch::empty` but every result is appended to an
// internal thread-local buffer. Must be paired with `end_ffi_alloc_record`.
// Caller must currently be in `kPassthrough` (no nested record/replay).
void begin_ffi_alloc_record();

// Move the recording out and return to `kPassthrough`. Must be paired with
// a prior `begin_ffi_alloc_record`.
std::vector<torch::Tensor> end_ffi_alloc_record();

// Switch to replay mode using a caller-owned recording. The hook will pop
// `recorded->at(i)` for the i-th request (with shape / dtype CHECKs). The
// caller MUST keep `*recorded` alive until `end_ffi_alloc_replay()`.
// Caller must currently be in `kPassthrough`.
void begin_ffi_alloc_replay(const std::vector<torch::Tensor>* recorded);

// Return to `kPassthrough` and detach from the recording pointer.
void end_ffi_alloc_replay();

// Current thread-local mode (mostly for diagnostics).
FfiAllocMode get_ffi_alloc_mode();

inline void bind_tvmffi_stream_to_current_torch_stream(
    const torch::Device& device) {
  if (is_torch_musa_device(device)) {
    sync_current_musa_stream(device);
    bind_musa_tvmffi_stream(device);
    return;
  }

  const c10::cuda::CUDAStream cur =
      c10::cuda::getCurrentCUDAStream(device.index());
  void* const stream = reinterpret_cast<void*>(cur.stream());
  if (stream == nullptr) {
    LOG(WARNING) << "[tvmffi.stream] current torch stream handle is null on "
                 << device;
  }

  constexpr int32_t kDlCuda = 2;
  constexpr int32_t kDlExtDev = 12;
  for (const int32_t device_type : {kDlCuda, kDlExtDev}) {
    void* original_stream = nullptr;
    const int rc = TVMFFIEnvSetStream(device_type, device.index(), stream,
                                      &original_stream);
    if (rc != 0) {
      LOG(WARNING) << "[tvmffi.stream] failed to set stream, rc=" << rc
                   << " dev_type=" << device_type << " dev=" << device.index();
    }
  }
}
}  // namespace xllm::kernel::cuda
