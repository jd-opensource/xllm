#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "cuda.h"

namespace xllm::kernel::cuda {

void prefill_reshape_and_cache(
    torch::Tensor proj_k,  // [shared_len, kv_heads, head_dim]
    torch::Tensor proj_v,  // [shared_len, kv_heads, head_dim]
    torch::Tensor
        shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
    torch::Tensor shared_v_cache) {
  int64_t shared_len = proj_k.size(0);
  shared_k_cache = shared_k_cache.slice(0, 0, shared_len);
  shared_v_cache = shared_v_cache.slice(0, 0, shared_len);
  shared_k_cache.copy_(proj_k);
  shared_v_cache.copy_(proj_v);
}

}  // namespace xllm::kernel::cuda
