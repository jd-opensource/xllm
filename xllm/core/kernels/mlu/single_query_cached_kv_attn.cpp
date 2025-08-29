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

#include "cnnl_extra.h"
#include "torch_mlu/csrc/aten/cnnl/cnnlHandle.h"
#include "torch_mlu/csrc/framework/core/mlu_guard.h"
#include "torch_ops_api.h"
#include "utils.h"

namespace xllm::mlu {

void single_query_cached_kv_attn(
    const torch::Tensor& q_ori,
    const torch::Tensor& k_cache,
    const torch::Tensor& output,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,  // [batch]
    const c10::optional<torch::Tensor>& v_cache,
    const c10::optional<torch::Tensor>& output_lse,
    const c10::optional<torch::Tensor>& q_quant_scale,
    const c10::optional<torch::Tensor>& k_cache_quant_scale,
    const c10::optional<torch::Tensor>& v_cache_quant_scale,
    const c10::optional<torch::Tensor>& out_quant_scale,
    const c10::optional<torch::Tensor>&
        alibi_slopes,  // [bs, head_num] or [head_num]
    const std::string& compute_dtype,
    int64_t max_context_len,
    int64_t windows_size_left,
    int64_t windows_size_right,
    double softmax_scale,
    bool return_lse,
    int64_t kv_cache_quant_bit_size) {
  // Check tensor type and tensor device.
  cnnlQuantizeScheme_t cache_quant_layout = CNNL_QUANTIZE_NONE;
  cnnlQuantizeScheme_t q_quant_layout = CNNL_QUANTIZE_NONE;
  cnnlQuantizeScheme_t qk_quant_layout = CNNL_QUANTIZE_NONE;
  bool is_kv_quant = k_cache_quant_scale.has_value();
  bool has_alibi = alibi_slopes.has_value();
  bool has_v = v_cache.has_value();

  // Check Tensor Shape
  int batch = q_ori.size(0);
  int seq_q = q_ori.size(1);
  int k_head_num = k_cache.size(1);
  int head_num = q_ori.size(2);
  int qk_head_size = q_ori.size(3);
  int max_num_block_per_seq = block_tables.size(-1);
  int num_blocks = k_cache.size(0);
  int block_size = k_cache.size(2);
  int cache_head_dim = k_cache.size(-1);
  int v_head_size = output.size(-1);
  TORCH_CHECK(windows_size_right < 0,
              "only support windows_size_right < 0 currently.");
  TORCH_CHECK(head_num % k_head_num == 0,
              "num_heads need be mutiple of num_kv_heads.");
  TORCH_CHECK(kv_cache_quant_bit_size == 4 || kv_cache_quant_bit_size == 8 ||
                  kv_cache_quant_bit_size == -1,
              "illegal quant bit size, only support 4, 8 or -1.");
  TORCH_CHECK(block_tables.dim() == 2 || block_tables.dim() == 3,
              "block_tables support 2-dim or 3-dm");
  if (block_tables.dim() == 2) {
    CHECK_SHAPE(block_tables, batch, max_num_block_per_seq)
  } else {
    CHECK_SHAPE(block_tables, batch, k_head_num, max_num_block_per_seq);
  }
  if (kv_cache_quant_bit_size == 4) {
    // int4 kv not support kv overlap
    cache_head_dim = qk_head_size;
    CHECK_SHAPE(
        k_cache, num_blocks, k_head_num, block_size, cache_head_dim / 2);
  } else {
    CHECK_SHAPE(k_cache, num_blocks, k_head_num, block_size, cache_head_dim);
  }
  TORCH_CHECK(q_ori.stride(-1) == 1 && q_ori.stride(-2) == qk_head_size,
              "q last two dim need be contiguous.");
  TORCH_CHECK(output.stride(-1) == 1 && output.stride(-2) == v_head_size,
              "output last two dim need be contiguous.")
  checkTensorContiguous(
      "output_lse, k_cache, v_cache, context_lens, k_cache_quant_scale, "
      "v_cache_quant_scale, "
      "alibi_slopes must be contiguous.",
      output_lse,
      k_cache,
      v_cache,
      context_lens,
      k_cache_quant_scale,
      v_cache_quant_scale,
      alibi_slopes);
  if (has_v) {
    if (kv_cache_quant_bit_size == 4) {
      int v_cache_len = PAD_UP_DIV(block_size, 2);
      CHECK_SHAPE(
          v_cache.value(), num_blocks, k_head_num, v_cache_len, v_head_size);
    } else {
      CHECK_SHAPE(
          v_cache.value(), num_blocks, k_head_num, block_size, v_head_size);
    }
  }
  TORCH_CHECK(!out_quant_scale.has_value(), "out can't be quant.");
  checkTensorSameAttr<TensorAttr::DEVICE>(q_ori,
                                          k_cache,
                                          v_cache,
                                          output,
                                          block_tables,
                                          context_lens,
                                          q_quant_scale,
                                          k_cache_quant_scale,
                                          v_cache_quant_scale,
                                          out_quant_scale,
                                          output_lse,
                                          alibi_slopes);
  if (q_quant_scale.has_value()) {
    TORCH_CHECK(is_kv_quant, "k/v must be quant when q is quant");
    if (q_quant_scale.value().dim() == 1 &&
        q_quant_scale.value().size(0) == 1) {
      // q/k/v fp8, per-tensor
      CHECK_SHAPE(k_cache_quant_scale.value(), 1);
      q_quant_layout = CNNL_QUANTIZE_PER_TENSOR;
      cache_quant_layout = CNNL_QUANTIZE_PER_TENSOR;
      qk_quant_layout = CNNL_QUANTIZE_PER_TENSOR;
    } else {
      // q/k/v int8, q/k per-token
      CHECK_SHAPE(q_quant_scale.value(), batch, seq_q, head_num);
      CHECK_SHAPE(
          k_cache_quant_scale.value(), num_blocks, k_head_num, block_size);
      q_quant_layout = CNNL_QUANTIZE_PER_TOKEN;
      qk_quant_layout = CNNL_QUANTIZE_PER_TOKEN;
      cache_quant_layout = CNNL_QUANTIZE_PER_TOKEN;
    }
  } else if (is_kv_quant) {
    TORCH_CHECK(k_cache_quant_scale.value().dim() >= 2 &&
                    k_cache_quant_scale.value().dim() <= 4,
                "k_cache_quant_scale must be 2d or 3d or 4d.");
    if (has_v) {
      TORCH_CHECK(v_cache_quant_scale.has_value() &&
                      k_cache_quant_scale.value().dim() ==
                          v_cache_quant_scale.value().dim(),
                  "the dim of k_cache_quant_scale and v_cache_quant_scale must "
                  "be euqal.");
    }
    if (k_cache_quant_scale.value().dim() == 2) {
      CHECK_SHAPE(k_cache_quant_scale.value(), k_head_num, cache_head_dim);
      if (v_cache_quant_scale.has_value())
        CHECK_SHAPE(v_cache_quant_scale.value(), k_head_num, v_head_size);
      cache_quant_layout = CNNL_QUANTIZE_PER_CHANNEL;
    } else if (k_cache_quant_scale.value().dim() == 3) {
      CHECK_SHAPE(
          k_cache_quant_scale.value(), num_blocks, k_head_num, block_size);
      if (v_cache_quant_scale.has_value())
        CHECK_SHAPE(
            v_cache_quant_scale.value(), num_blocks, k_head_num, block_size);
      cache_quant_layout = CNNL_QUANTIZE_PER_TOKEN;
    } else {
      CHECK_SHAPE(
          k_cache_quant_scale.value(), num_blocks, k_head_num, block_size, 1);
      if (v_cache_quant_scale.has_value())
        CHECK_SHAPE(
            v_cache_quant_scale.value(), num_blocks, k_head_num, block_size, 1);
      cache_quant_layout = CNNL_QUANTIZE_PER_TOKEN;
    }
  }
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32 ||
                  block_tables.scalar_type() == torch::kLong,
              "block_tables type need be torch::kInt32 or torch::kLong.");
  // Check context_lens
  TORCH_CHECK(context_lens.dtype() == torch::kInt32,
              "context_lens type need be torch::kInt32.");

  if (has_alibi) {
    CHECK_SHAPE(alibi_slopes.value(), batch, head_num);
  }
  if (return_lse) {
    TORCH_CHECK(seq_q == 1, "return lse only support seq_q = 1 currently.");
    CHECK_SHAPE(output_lse.value(), batch, head_num, seq_q);
  }

  // Convert torch tensor to tensor descs
  auto descs = createTensorDescs({q_ori,
                                  k_cache,
                                  v_cache.value_or(at::Tensor()),
                                  k_cache_quant_scale.value_or(at::Tensor()),
                                  v_cache_quant_scale.value_or(at::Tensor()),
                                  context_lens,
                                  block_tables,
                                  alibi_slopes.value_or(at::Tensor()),
                                  output,
                                  output_lse.value_or(at::Tensor()),
                                  q_quant_scale.value_or(at::Tensor()),
                                  out_quant_scale.value_or(at::Tensor())});
  if (kv_cache_quant_bit_size == 4) {
    cnnlDataType_t data_type = CNNL_DTYPE_INT4X2;
    // k_cache
    CNNL_CHECK_FATAL(cnnlSetTensorDescriptor_v2(descs[1].get(),
                                                CNNL_LAYOUT_ARRAY,
                                                data_type,
                                                k_cache.sizes().size(),
                                                k_cache.sizes().data()));
    // v_cache
    if (has_v)
      CNNL_CHECK_FATAL(
          cnnlSetTensorDescriptor_v2(descs[2].get(),
                                     CNNL_LAYOUT_ARRAY,
                                     data_type,
                                     v_cache.value().sizes().size(),
                                     v_cache.value().sizes().data()));
  }
  cnnlDataType_t cnnl_compute_dtype =
      compute_dtype == "float"  ? CNNL_DTYPE_FLOAT
      : compute_dtype == "half" ? CNNL_DTYPE_HALF
                                : CNNL_DTYPE_BFLOAT16;

  // Get current handle.
  const torch_mlu::mlu::MLUGuard device_guard(q_ori.device());
  cnnlHandle_t handle = torch_mlu::getCurrentHandle();
  // Get workspace size and malloc workspace.
  size_t workspace_size = 0;
  cnnlSingleQueryCachedKVAttnDescriptor_t att_desc = nullptr;
  CNNL_CHECK_FATAL(
      cnnlGetSingleQueryCachedKVAttnWorkspaceSize_v2(handle,
                                                     att_desc,
                                                     descs[0].get(),
                                                     descs[1].get(),
                                                     descs[2].get(),
                                                     (int)max_context_len,
                                                     &workspace_size));
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             q_ori.options().dtype(at::kByte));

  CNNL_CHECK_FATAL(
      cnnlSingleQueryCachedKVAttn_v6(handle,
                                     att_desc,
                                     descs[0].get(),
                                     getAtTensorPtr(q_ori),
                                     descs[1].get(),
                                     getAtTensorPtr(k_cache),
                                     descs[2].get(),
                                     getAtTensorPtr(v_cache),
                                     descs[10].get(),
                                     getAtTensorPtr(q_quant_scale),
                                     descs[3].get(),
                                     getAtTensorPtr(k_cache_quant_scale),
                                     descs[4].get(),
                                     getAtTensorPtr(v_cache_quant_scale),
                                     nullptr,
                                     nullptr,
                                     descs[5].get(),
                                     getAtTensorPtr(context_lens),
                                     descs[6].get(),
                                     getAtTensorPtr(block_tables),
                                     descs[7].get(),
                                     getAtTensorPtr(alibi_slopes),
                                     q_quant_layout,
                                     cache_quant_layout,
                                     CNNL_QUANTIZE_NONE,
                                     qk_quant_layout,
                                     (int)max_context_len,
                                     windows_size_left,
                                     windows_size_right,
                                     qk_head_size,
                                     v_head_size,
                                     softmax_scale,
                                     return_lse,
                                     cnnl_compute_dtype,
                                     getAtTensorPtr(workspace),
                                     workspace_size,
                                     descs[8].get(),
                                     getAtTensorPtr(output),
                                     descs[11].get(),
                                     getAtTensorPtr(out_quant_scale),
                                     descs[9].get(),
                                     getAtTensorPtr(output_lse)));
}

}  // namespace xllm::mlu
