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

#include "attention.h"

#include "core/common/global_flags.h"
#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // Check if attention_mask is provided
  bool use_custom_mask = attn_metadata.attn_mask.defined();

  // Get block_size from k_cache if defined and has proper dimensions,
  // otherwise use a default value (for prefill without KV cache, e.g., LongCat)
  int64_t block_size = 1;  // Default value when KV cache is not initialized
  if (k_cache.defined() && k_cache.dim() >= 2) {
    block_size = k_cache.size(1);
  }

  // maybe we need to update shared attn state before execute attention,
  // currently we update flashinfer step_wise_attn_state_ at layer 0.
  bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  // When using a custom mask, we set causal=false for the plan because:
  // 1. The custom mask already encodes the causal constraint (lower triangular)
  // 2. The plan and run function must be consistent in their mask mode
  // 3. If plan uses causal=true but run uses CUSTOM mask mode, FlashInfer may
  //    ignore the custom mask in favor of its optimized causal path
  bool causal_for_plan = !use_custom_mask;

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        is_prefill ? xllm::kernel::cuda::determine_attention_backend(
                         /*pos_encoding_mode=*/0,
                         /*use_fp16_qk_reduction=*/false,
                         /*use_custom_mask=*/use_custom_mask)
                   : "fa2",
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size=*/block_size,
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/attn_metadata.enable_cuda_graph,
        /*causal=*/causal_for_plan,
        /*use_tensor_core=*/attn_metadata.use_tensor_core);
  }

  // Only reshape and store to cache if k_cache is properly initialized
  // For prefill without KV cache (e.g., LongCat text encoding), skip this step
  if (k_cache.defined() && k_cache.dim() >= 2) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  xllm::kernel::AttentionParams attention_params(attn_metadata);
  attention_params.query = query;
  attention_params.output = output;
  attention_params.output_lse = output_lse;
  // attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  // for flashinfer
  attention_params.float_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  attention_params.int_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  attention_params.page_locked_int_workspace_buffer =
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  // Pass attention_mask if provided (use_custom_mask implies
  // attn_mask.defined())
  if (use_custom_mask) {
    attention_params.mask = attn_metadata.attn_mask;
  }

  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";
  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    // Eager PyTorch attention fallback when custom mask (LongCat
    // padding+causal). FlashInfer custom mask gives wrong token-0 output; eager
    // matches diffusers.
    if (use_custom_mask) {
      torch::Tensor m = attn_metadata.attn_mask;
      if (m.device() != query.device()) m = m.to(query.device());
      if (!m.is_floating_point()) m = m.to(torch::kFloat32);
      int64_t T = query.size(0);
      CHECK_EQ(m.size(0), T) << "[eager attention] mask length " << m.size(0)
                             << " != query seq len " << T;
      auto device = query.device();
      auto causal = torch::tril(torch::ones(
          {T, T},
          torch::TensorOptions().dtype(torch::kFloat32).device(device)));
      auto pad2d = m.unsqueeze(0).expand({T, T});
      auto combined = (causal * pad2d).to(torch::kFloat32);
      const float mask_val = -std::numeric_limits<float>::infinity();
      auto add_mask = torch::where(combined > 0.5f,
                                   torch::zeros_like(combined),
                                   torch::full_like(combined, mask_val));
      int64_t g = num_heads_ / num_kv_heads_;
      // [T,K,D] -> [T,K,D,1] -> [T,K,D,g] -> permute to [T,K,g,D] -> [T,K*g,D].
      // Head h = kv_head k, replicate r: h = k*g + r; each head gets full D
      // dims.
      auto Kg = key.unsqueeze(3).expand({-1, -1, -1, g});
      auto Vg = value.unsqueeze(3).expand({-1, -1, -1, g});
      torch::Tensor Kr =
          Kg.permute({0, 1, 3, 2}).reshape({-1, num_heads_, head_size_});
      torch::Tensor Vr =
          Vg.permute({0, 1, 3, 2}).reshape({-1, num_heads_, head_size_});
      auto Qf = query.to(torch::kFloat32);
      auto Kf = Kr.to(torch::kFloat32);
      // scores[t,h,j] = sum_d Q[t,h,d]*K[j,h,d]. Use
      // (T,H,1,D)*(1,H,T,D)->(T,H,T,D)->sum(-1).
      auto Kf_1HTD = Kf.permute({1, 0, 2}).unsqueeze(0);  // [1, H, T, D]
      auto scores = (Qf.unsqueeze(2) * Kf_1HTD).sum(-1) * scale_;
      scores = scores + add_mask.unsqueeze(1);
      // Match diffusers: softmax in float32, cast attn to query dtype; attn @ V
      // in bf16.
      auto attn = torch::softmax(scores.to(torch::kFloat32), -1)
                      .to(query.scalar_type());
      auto out = torch::einsum("thj,jhd->thd", {attn, Vr}).contiguous();
      auto result = out.view({-1, num_heads_ * head_size_});
      return {result, output_lse};
    }
    xllm::kernel::batch_prefill(attention_params);
  } else {
    attention_params.query = query;
    attention_params.output = output;
    attention_params.k_cache = k_cache;
    attention_params.v_cache = v_cache;

    xllm::kernel::batch_decode(attention_params);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm