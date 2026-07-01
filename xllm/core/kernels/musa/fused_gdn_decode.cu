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

// Single-launch fused gated-delta-net decode kernel (T=1 per sequence).
// Ported in semantics from sglang_qwen35
//   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
// with adaptations for xLLM/MUSA:
//   * State cache uses xLLM storage layout [pool, Hv, V, K] (V outer, K
//     innermost). chunk_gated_delta_rule produces FLA-layout state [B, Hv, K,
//     V] and the layer writes it back with .transpose(-1, -2) when
//     use_fla_ssm_state_layout() == false (Qwen3.5's default), which means the
//     contiguous storage in the pool is K-innermost. The state base offset is
//       slot * Hv * V * K + hv * V * K + v * K + k
//     i.e. stride-to-next-K is 1; stride-to-next-V is K. Reading with the
//     opposite stride pair effectively transposes the state and produces
//     garbage decode output even though the math compiles cleanly.
//   * State dtype is always fp32 in the kernel. Qwen3.5 / Qwen3-Next set
//     mamba_ssm_dtype="float32" so the SSM cache is allocated as fp32 even when
//     the rest of the model is bf16; the launcher CHECKs this. Mate decode
//     casts state to fp32 internally for the same reason -- we just require
//     it directly on the tensor so reads/writes are in-place without a copy.
//   * Inputs q/k/v are read directly from `mixed_qkv` [B, D] via strided
//     reads (D = 2 * Hk * K + Hv * V); no per-step contiguous() copy.
//   * Gating math (-exp(A_log) * softplus(a + dt_bias), sigmoid(b)) is
//     fused into the kernel; identical formulation to gdn_gating.cu so the
//     softplus saturates linearly above threshold for numerical stability.
//   * Block-level reductions use shared memory + __syncthreads() instead of
//     warp shuffles (MUSA-safe across SIMD widths). Accumulate in fp32,
//     store bf16/fp16.
//   * State is read twice (once for kv/dot_qH_v accumulation, once for the
//     write-back pass) to avoid a per-thread float h[K] register array.
//     Removes register pressure at the cost of one extra HBM read per
//     (k, v) element; coalesced because adjacent threads own adjacent v.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

#include "core/kernels/musa/cuda_ops_api.h"
#include "core/kernels/musa/mate_gdn_ops.h"
#include "core/kernels/param.h"

namespace xllm::kernel::cuda {

namespace {

// Largest K (head_k_dim) supported by the static shared buffers below. The
// Qwen3-Next / Qwen3.5 config uses K = V = 128; bumping this only costs static
// shared memory. Launcher CHECKs that the runtime K/V both fit.
constexpr int kFusedGdnDecodeMaxKV = 256;

template <typename scalar_t>
__global__ void fused_gdn_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    int64_t mixed_qkv_row_stride,
    float* __restrict__ state,
    const float* __restrict__ A_log_f32,
    const scalar_t* __restrict__ a,
    const float* __restrict__ dt_bias_f32,
    const scalar_t* __restrict__ b,
    const int32_t* __restrict__ state_indices,
    scalar_t* __restrict__ output,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t qk_cols,
    int64_t v_cols,
    float scale,
    float softplus_beta,
    float softplus_threshold) {
  const int batch = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;

  const int64_t slot = static_cast<int64_t>(state_indices[batch]);
  // Padded slot: skip whole block. All threads must reach this branch before
  // any __syncthreads is issued below.
  if (slot < 0) {
    return;
  }

  // GQA: group several v-heads to one k/q-head.
  const int group = static_cast<int>(num_v_heads / num_k_heads);
  const int hk = hv / group;

  const int K = static_cast<int>(head_k_dim);
  const int V = static_cast<int>(head_v_dim);

  __shared__ float q_sh[kFusedGdnDecodeMaxKV];
  __shared__ float k_sh[kFusedGdnDecodeMaxKV];
  __shared__ float v_sh[kFusedGdnDecodeMaxKV];
  __shared__ float reduce_buf[kFusedGdnDecodeMaxKV];
  __shared__ float g_exp_sh;
  __shared__ float beta_sh;
  __shared__ float dot_qk_sh;

  const scalar_t* qkv_row = mixed_qkv +
                            static_cast<int64_t>(batch) * mixed_qkv_row_stride;
  const int64_t q_base = static_cast<int64_t>(hk) * K;
  const int64_t k_base = qk_cols + static_cast<int64_t>(hk) * K;
  const int64_t v_base = 2 * qk_cols + static_cast<int64_t>(hv) * V;

  for (int k = tid; k < K; k += block_threads) {
    q_sh[k] = static_cast<float>(qkv_row[q_base + k]);
    k_sh[k] = static_cast<float>(qkv_row[k_base + k]);
  }
  for (int v = tid; v < V; v += block_threads) {
    v_sh[v] = static_cast<float>(qkv_row[v_base + v]);
  }
  __syncthreads();

  // L2-norm over K for q and k (eps = 1e-6), then q *= scale. Matches
  // sglang fused kernel L176-180. Use shared-memory tree reduction.
  float local_q2 = 0.f;
  float local_k2 = 0.f;
  for (int k = tid; k < K; k += block_threads) {
    local_q2 += q_sh[k] * q_sh[k];
    local_k2 += k_sh[k] * k_sh[k];
  }
  reduce_buf[tid] = local_q2;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  const float q_norm_inv = rsqrtf(reduce_buf[0] + 1e-6f);
  __syncthreads();
  reduce_buf[tid] = local_k2;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  const float k_norm_inv = rsqrtf(reduce_buf[0] + 1e-6f);
  __syncthreads();

  for (int k = tid; k < K; k += block_threads) {
    q_sh[k] = q_sh[k] * q_norm_inv * scale;
    k_sh[k] = k_sh[k] * k_norm_inv;
  }

  // Gating: g = -exp(A_log[hv]) * softplus(a + dt_bias; beta, threshold),
  // beta_gate = sigmoid(b). Identical formulation to gdn_gating.cu L50-55,
  // softplus saturates linearly above threshold for stability.
  if (tid == 0) {
    const float a_val = static_cast<float>(
        a[static_cast<int64_t>(batch) * num_v_heads + hv]);
    const float b_val = static_cast<float>(
        b[static_cast<int64_t>(batch) * num_v_heads + hv]);
    const float pre = a_val + dt_bias_f32[hv];
    const float bx = softplus_beta * pre;
    const float sp =
        (bx > softplus_threshold) ? pre : (log1pf(expf(bx)) / softplus_beta);
    const float g = -expf(A_log_f32[hv]) * sp;
    g_exp_sh = expf(g);
    beta_sh = 1.f / (1.f + expf(-b_val));
  }
  __syncthreads();

  // dot_qk = sum_k q_sh[k] * k_sh[k], independent of v -- compute once
  // per block. Required by the closed-form output o[v] = exp(g)*dot_qH_v +
  // delta * dot_qk derived from o[v] = sum_k h_new[k,v] * q_sh[k].
  float local_qk = 0.f;
  for (int k = tid; k < K; k += block_threads) {
    local_qk += q_sh[k] * k_sh[k];
  }
  reduce_buf[tid] = local_qk;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    dot_qk_sh = reduce_buf[0];
  }
  __syncthreads();

  const float g_exp = g_exp_sh;
  const float beta_val = beta_sh;
  const float dot_qk = dot_qk_sh;

  // Each thread owns one v column. Storage layout is [pool, Hv, V, K] with K
  // innermost, so for fixed v the K state slots are contiguous in memory.
  // First pass accumulates kv = sum_k state[v,k] * k_sh[k] and dot_qH_v =
  // sum_k state[v,k] * q_sh[k]; second pass writes
  //   h_new[v,k] = exp(g) * state[v,k] + k_sh[k] * delta
  // back to the cache. Adjacent threads index adjacent v -> warp lanes stride
  // by K (not coalesced); this matches the mate decode access pattern and is
  // acceptable for correctness. The per-thread inner k-loop is contiguous so
  // L1/L2 hit rate stays high.
  if (tid < V) {
    const int64_t state_base =
        ((slot * num_v_heads + hv) * static_cast<int64_t>(V) +
         static_cast<int64_t>(tid)) *
        static_cast<int64_t>(K);

    float kv = 0.f;
    float dot_qH_v = 0.f;
    for (int k = 0; k < K; ++k) {
      const float s = state[state_base + static_cast<int64_t>(k)];
      kv += s * k_sh[k];
      dot_qH_v += s * q_sh[k];
    }

    const float delta = (v_sh[tid] - g_exp * kv) * beta_val;
    const float o_v = g_exp * dot_qH_v + delta * dot_qk;

    const int64_t out_off =
        (static_cast<int64_t>(batch) * num_v_heads + hv) * V +
        static_cast<int64_t>(tid);
    output[out_off] = static_cast<scalar_t>(o_v);

    for (int k = 0; k < K; ++k) {
      const int64_t off = state_base + static_cast<int64_t>(k);
      const float s_old = state[off];
      state[off] = g_exp * s_old + k_sh[k] * delta;
    }
  }
}

template <typename scalar_t>
void launch(const torch::Tensor& mixed_qkv,
            torch::Tensor& state,
            const torch::Tensor& A_log_f32,
            const torch::Tensor& a,
            const torch::Tensor& dt_bias_f32,
            const torch::Tensor& b,
            const torch::Tensor& state_indices_i32,
            torch::Tensor& output,
            int64_t num_k_heads,
            int64_t num_v_heads,
            int64_t head_k_dim,
            int64_t head_v_dim,
            int64_t qk_cols,
            int64_t v_cols,
            int64_t batch_size,
            float scale,
            float softplus_beta,
            float softplus_threshold,
            cudaStream_t stream) {
  const int block_threads = static_cast<int>(
      head_v_dim < head_k_dim ? head_k_dim : head_v_dim);
  const dim3 grid(static_cast<unsigned int>(batch_size),
                  static_cast<unsigned int>(num_v_heads),
                  1);
  fused_gdn_decode_kernel<scalar_t><<<grid, block_threads, 0, stream>>>(
      mixed_qkv.data_ptr<scalar_t>(),
      mixed_qkv.stride(0),
      state.data_ptr<float>(),
      A_log_f32.data_ptr<float>(),
      a.data_ptr<scalar_t>(),
      dt_bias_f32.data_ptr<float>(),
      b.data_ptr<scalar_t>(),
      state_indices_i32.data_ptr<int32_t>(),
      output.data_ptr<scalar_t>(),
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      qk_cols,
      v_cols,
      scale,
      softplus_beta,
      softplus_threshold);
}

}  // namespace

torch::Tensor fused_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params) {
  auto mixed_qkv = params.mixed_qkv.contiguous();
  CHECK(mixed_qkv.dim() == 2)
      << "fused GDN decode expects mixed_qkv [B, D], got "
      << mixed_qkv.dim() << "-D";

  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t num_k_heads = params.num_k_heads;
  const int64_t num_v_heads = params.num_v_heads;
  const int64_t head_k_dim = params.head_k_dim;
  const int64_t head_v_dim = params.head_v_dim;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;

  CHECK(mixed_qkv.size(1) == 2 * qk_cols + v_cols)
      << "fused GDN decode mixed_qkv last dim mismatch: got "
      << mixed_qkv.size(1) << " expected " << (2 * qk_cols + v_cols);
  CHECK(head_k_dim > 0 && head_k_dim <= kFusedGdnDecodeMaxKV)
      << "fused GDN decode head_k_dim " << head_k_dim << " out of range (1, "
      << kFusedGdnDecodeMaxKV << "]";
  CHECK(head_v_dim > 0 && head_v_dim <= kFusedGdnDecodeMaxKV)
      << "fused GDN decode head_v_dim " << head_v_dim << " out of range (1, "
      << kFusedGdnDecodeMaxKV << "]";
  CHECK(num_v_heads % num_k_heads == 0)
      << "fused GDN decode requires num_v_heads divisible by num_k_heads (GQA)";
  CHECK(params.state.dim() == 4)
      << "fused GDN decode expects state [pool, Hv, V, K], got "
      << params.state.dim() << "-D";
  // Storage layout: [pool, Hv, V_outer, K_inner]. Qwen3.5 writes the cache
  // through last_recurrent_state.transpose(-1, -2) (use_fla_ssm_state_layout
  // is false), so the contiguous innermost axis is head_k_dim. K and V are
  // both 128 for Qwen3.5 so the dim CHECK alone cannot catch a swap; the
  // kernel relies on the documented storage order.
  CHECK(params.state.size(1) == num_v_heads &&
        params.state.size(2) == head_v_dim &&
        params.state.size(3) == head_k_dim)
      << "fused GDN decode state shape mismatch with head dims (expected [_, "
      << num_v_heads << ", " << head_v_dim << ", " << head_k_dim << "])";
  CHECK(params.state.is_contiguous())
      << "fused GDN decode requires contiguous state cache";
  // Qwen3.5 / Qwen3-Next allocate the SSM cache as fp32 (mamba_ssm_dtype). We
  // update it in place, so require fp32 directly rather than casting & copying
  // back -- mismatches indicate an unsupported config and would silently lose
  // precision.
  CHECK(params.state.scalar_type() == torch::kFloat32)
      << "fused GDN decode requires fp32 state cache (got "
      << params.state.scalar_type() << ")";

  auto a = params.a;
  if (a.dim() == 1) {
    a = a.unsqueeze(0);
  }
  auto b = params.b;
  if (b.dim() == 1) {
    b = b.unsqueeze(0);
  }
  CHECK(a.dim() == 2 && b.dim() == 2)
      << "fused GDN decode expects a/b shaped [B, Hv]";
  a = a.contiguous();
  b = b.contiguous();

  auto A_log_f32 = params.A_log.to(torch::kFloat32).contiguous();
  auto dt_bias_f32 = params.dt_bias.to(torch::kFloat32).contiguous();
  auto state_indices_i32 =
      params.state_indices.to(torch::kInt32).contiguous();

  torch::Tensor output;
  if (params.decode_output.has_value() && params.decode_output.value().defined()) {
    output = params.decode_output.value();
    CHECK(output.is_contiguous())
        << "fused GDN decode requires contiguous decode_output";
  } else {
    output = torch::empty({batch_size, num_v_heads, head_v_dim},
                          mixed_qkv.options());
  }

  const float scale = static_cast<float>(params.scale);
  // Softplus parameters mirror gdn_gating.cu defaults / sglang kernel defaults.
  const float softplus_beta = 1.0f;
  const float softplus_threshold = 20.0f;

  const at::cuda::OptionalCUDAGuard guard(device_of(mixed_qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mixed_qkv.scalar_type() == torch::kBFloat16) {
    launch<at::BFloat16>(mixed_qkv,
                         params.state,
                         A_log_f32,
                         a,
                         dt_bias_f32,
                         b,
                         state_indices_i32,
                         output,
                         num_k_heads,
                         num_v_heads,
                         head_k_dim,
                         head_v_dim,
                         qk_cols,
                         v_cols,
                         batch_size,
                         scale,
                         softplus_beta,
                         softplus_threshold,
                         stream);
  } else if (mixed_qkv.scalar_type() == torch::kFloat16) {
    launch<at::Half>(mixed_qkv,
                     params.state,
                     A_log_f32,
                     a,
                     dt_bias_f32,
                     b,
                     state_indices_i32,
                     output,
                     num_k_heads,
                     num_v_heads,
                     head_k_dim,
                     head_v_dim,
                     qk_cols,
                     v_cols,
                     batch_size,
                     scale,
                     softplus_beta,
                     softplus_threshold,
                     stream);
  } else {
    LOG(FATAL) << "fused GDN decode: unsupported dtype "
               << mixed_qkv.scalar_type();
  }

  return output;
}

}  // namespace xllm::kernel::cuda
