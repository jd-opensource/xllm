/* Copyright 2025 The vLLM Authors and The xLLM Authors. All Rights Reserved.

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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cuda_ops_api.h"
#include "device_utils.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu

namespace {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = *(cos_ptr + x_index);
    sin = *(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = *(cos_ptr + x_index / 2);
    sin = *(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

// Template the position index type so callers may pass either int32_t
// (xllm's graph-mode persistent positions buffer) or int64_t (eager-mode
// positions and sglang's convention). The kernel only ever widens the value
// to int64_t internally before computing the cache offset, so int32 indices
// are safe as long as positions stay within int32 range (true for
// max_position_embeddings well below 2^31).
template <typename scalar_t, typename idx_t, bool IS_NEOX>
__global__ void XLLM_KERNEL_ATTR(512) rotary_embedding_kernel(
    const idx_t* __restrict__ positions,  // [batch_size, seq_len] or
                                          // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
    // head_size] or [num_tokens, num_kv_heads,
    // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2,
                                                 // rot_dim // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = static_cast<int64_t>(positions[token_idx]);
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(query,
                                            key,
                                            cache_ptr,
                                            head_size,
                                            num_heads,
                                            num_kv_heads,
                                            rot_dim,
                                            token_idx,
                                            query_stride,
                                            key_stride,
                                            head_stride);
}
}  // namespace

namespace xllm::kernel::cuda {

// flashinfer rope ops
// void apply_rope_pos_ids_cos_sin_cache(torch::Tensor q,
//                                       torch::Tensor k,
//                                       torch::Tensor cos_sin_cache,
//                                       torch::Tensor pos_ids,
//                                       bool interleave) {
//   const int64_t head_dim = cos_sin_cache.size(-1) / 2;
//   q = q.view({q.size(0), -1, head_dim});
//   k = k.view({k.size(0), -1, head_dim});

//   FunctionFactory::get_instance().rope_func("rope").call(
//       q, k, q, k, cos_sin_cache, pos_ids, interleave);
// }

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    // int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t head_size = cos_sin_cache.size(-1);
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  CHECK(positions_ndim == 1 || positions_ndim == 2)
      << "positions must have shape [num_tokens] or [batch_size, seq_len]";

  if (positions_ndim == 1) {
    CHECK(query.size(0) == positions.size(0) &&
          (!key.has_value() || key->size(0) == positions.size(0)))
        << "query, key and positions must have the same number of tokens";
  }
  if (positions_ndim == 2) {
    CHECK(query.size(0) == positions.size(0) &&
          (!key.has_value() || key->size(0) == positions.size(0)) &&
          query.size(1) == positions.size(1) &&
          (!key.has_value() || key->size(1) == positions.size(1)))
        << "query, key and positions must have the same batch_size and seq_len";
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  CHECK(query_hidden_size % head_size == 0);
  CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Accept both int32 and int64 positions. Eager-mode callers (qwen2,
  // deepseek_v4) pre-cast to int64 via positions.to(torch::kInt64), but the
  // graph-mode persistent positions buffer in CudaGraphPersistentParam stays
  // int32 to share dtype with the metadata fast-path kernel. Inside a CUDA
  // graph capture region the .to(kInt64) cast would have to allocate a new
  // tensor, which the caching allocator forbids during capture; dispatching
  // on the input dtype here keeps the kernel graph-safe without forcing the
  // persistent buffer (or the metadata copy kernel) to widen to int64.
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64)
      << "positions must be int32 or int64, got " << positions.scalar_type();
  const bool positions_is_int64 = positions.scalar_type() == torch::kInt64;
  DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "apply_rope_pos_ids_cos_sin_cache", [&] {
        scalar_t* query_ptr = query.data_ptr<scalar_t>();
        scalar_t* key_ptr =
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
        const scalar_t* cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        if (positions_is_int64) {
          const int64_t* pos_ptr = positions.data_ptr<int64_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int64_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          } else {
            rotary_embedding_kernel<scalar_t, int64_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          }
        } else {
          const int32_t* pos_ptr = positions.data_ptr<int32_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int32_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          } else {
            rotary_embedding_kernel<scalar_t, int32_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr,
                                             query_ptr,
                                             key_ptr,
                                             cache_ptr,
                                             rot_dim,
                                             query_stride,
                                             key_stride,
                                             head_stride,
                                             num_heads,
                                             num_kv_heads,
                                             head_size);
          }
        }
      });
}

// CUDA-graph-safe partial rotary embedding (in-place).
//
// Applies rotary embedding to the first `rotary_dim` elements of every
// (token, head) slot in `query` and `key`, leaving the remaining
// `head_size - rotary_dim` "pass-through" elements untouched. Replaces the
// libtorch chain in gdn_ops.cpp::partial_rotary_embedding which materialised
// q_rot/k_rot via `slice(-1, 0, rotary_dim).contiguous()` and then re-joined
// the pass-through suffix with `torch::cat({...rot, ...pass}, -1).reshape(...)`
// -- both `.contiguous()` and `torch::cat` invoke `at::empty`-class allocators
// which torch_musa 2.7.1 rejects during CUDA-graph capture.
//
// The underlying `rotary_embedding_kernel` already supports partial rotary
// natively: its inner loop runs over `num_heads * (rotary_dim / 2)` indices,
// touching only offsets `[0, rotary_dim)` within each head_size slot. We just
// expose a host wrapper that lets callers pass `head_size` explicitly (the
// public `rotary_embedding` derives both `head_size` and `rot_dim` from
// `cos_sin_cache.size(-1)`, which conflates them and is correct only for
// full rotary).
//
// Preconditions enforced via TORCH_CHECK:
//   * query / key: 2D `[num_tokens, num_heads * head_size]` (or higher-rank
//     where everything but `num_tokens` flattens to `num_heads * head_size`).
//     Both must be contiguous in the head/feature axis (`stride(-1) == 1`).
//   * cos_sin_cache: 2D `[max_position, rotary_dim]`, contiguous.
//   * positions: int32 or int64. Same shape contract as `rotary_embedding`.
//   * `rotary_dim <= head_size` and `rotary_dim` is even.
void partial_rotary_embedding_inplace(torch::Tensor& positions,
                                      torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      int64_t head_size,
                                      int64_t rotary_dim,
                                      bool is_neox) {
  TORCH_CHECK(head_size > 0, "partial_rotary_embedding_inplace: head_size must be > 0");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_size,
              "partial_rotary_embedding_inplace: 0 < rotary_dim <= head_size");
  TORCH_CHECK(rotary_dim % 2 == 0,
              "partial_rotary_embedding_inplace: rotary_dim must be even");
  TORCH_CHECK(cos_sin_cache.size(-1) == rotary_dim,
              "partial_rotary_embedding_inplace: cos_sin_cache last dim (",
              cos_sin_cache.size(-1), ") must equal rotary_dim (",
              rotary_dim, ")");

  const int64_t num_tokens = positions.numel();
  const int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");

  const int64_t query_hidden_size = query.numel() / num_tokens;
  const int64_t key_hidden_size = key.numel() / num_tokens;
  TORCH_CHECK(query_hidden_size % head_size == 0,
              "partial_rotary_embedding_inplace: query hidden_size must be "
              "divisible by head_size");
  TORCH_CHECK(key_hidden_size % head_size == 0,
              "partial_rotary_embedding_inplace: key hidden_size must be "
              "divisible by head_size");
  TORCH_CHECK(query.stride(-1) == 1 && key.stride(-1) == 1,
              "partial_rotary_embedding_inplace: query/key last dim must be "
              "contiguous (stride==1)");
  TORCH_CHECK(cos_sin_cache.is_contiguous(),
              "partial_rotary_embedding_inplace: cos_sin_cache must be contiguous");

  const int num_heads = static_cast<int>(query_hidden_size / head_size);
  const int num_kv_heads = static_cast<int>(key_hidden_size / head_size);
  TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
              "partial_rotary_embedding_inplace: num_heads must be a multiple "
              "of num_kv_heads");

  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.stride(seq_dim_idx);
  const int query_ndim = query.dim();
  // Mirror `rotary_embedding`: for an explicit per-head layout (one more dim
  // than positions) use the actual stride; for a flat [tokens, heads*head]
  // layout, heads occupy contiguous blocks of `head_size`.
  const int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(static_cast<unsigned>(num_tokens));
  const int rot_dim_i = static_cast<int>(rotary_dim);
  const int head_size_i = static_cast<int>(head_size);
  dim3 block(std::min<int64_t>(static_cast<int64_t>(num_heads) * (rot_dim_i / 2),
                                512));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(positions.scalar_type() == torch::kInt32 ||
                  positions.scalar_type() == torch::kInt64,
              "partial_rotary_embedding_inplace: positions must be int32 or "
              "int64, got ", positions.scalar_type());
  const bool positions_is_int64 = positions.scalar_type() == torch::kInt64;

  DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "partial_rotary_embedding_inplace", [&] {
        scalar_t* query_ptr = query.data_ptr<scalar_t>();
        scalar_t* key_ptr = key.data_ptr<scalar_t>();
        const scalar_t* cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        if (positions_is_int64) {
          const int64_t* pos_ptr = positions.data_ptr<int64_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int64_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr, query_ptr, key_ptr,
                                             cache_ptr, rot_dim_i,
                                             query_stride, key_stride,
                                             head_stride, num_heads,
                                             num_kv_heads, head_size_i);
          } else {
            rotary_embedding_kernel<scalar_t, int64_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr, query_ptr, key_ptr,
                                             cache_ptr, rot_dim_i,
                                             query_stride, key_stride,
                                             head_stride, num_heads,
                                             num_kv_heads, head_size_i);
          }
        } else {
          const int32_t* pos_ptr = positions.data_ptr<int32_t>();
          if (is_neox) {
            rotary_embedding_kernel<scalar_t, int32_t, true>
                <<<grid, block, 0, stream>>>(pos_ptr, query_ptr, key_ptr,
                                             cache_ptr, rot_dim_i,
                                             query_stride, key_stride,
                                             head_stride, num_heads,
                                             num_kv_heads, head_size_i);
          } else {
            rotary_embedding_kernel<scalar_t, int32_t, false>
                <<<grid, block, 0, stream>>>(pos_ptr, query_ptr, key_ptr,
                                             cache_ptr, rot_dim_i,
                                             query_stride, key_stride,
                                             head_stride, num_heads,
                                             num_kv_heads, head_size_i);
          }
        }
      });
}

}  // namespace xllm::kernel::cuda
