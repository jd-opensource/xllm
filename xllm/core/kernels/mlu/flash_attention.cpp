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

void flash_attention(const at::Tensor& q,
                     const at::Tensor& k,
                     const at::Tensor& v,
                     const at::Tensor& output,
                     const c10::optional<at::Tensor>& output_lse,
                     const c10::optional<at::Tensor>& cu_seq_lens_q,
                     const c10::optional<at::Tensor>& cu_seq_lens_kv,
                     const c10::optional<at::Tensor>& alibi_slope,
                     const c10::optional<at::Tensor>& attn_bias,
                     const c10::optional<at::Tensor>& q_quant_scale,
                     const c10::optional<at::Tensor>& k_quant_scale,
                     const c10::optional<at::Tensor>& v_quant_scale,
                     const c10::optional<at::Tensor>& out_quant_scale,
                     const c10::optional<at::Tensor>& block_tables,
                     const int64_t max_seq_len_q,
                     const int64_t max_seq_len_kv,
                     const double softmax_scale,
                     const bool is_causal,
                     const int64_t window_size_left,
                     const int64_t window_size_right,
                     const std::string& compute_dtype,
                     bool return_lse) {
  TORCH_CHECK(compute_dtype == "float" || compute_dtype == "half" ||
                  compute_dtype == "bfloat16",
              "compute_dtype must be 'float', 'half' or 'bfloat16'.");
  bool has_block_table = block_tables.has_value();
  // 3d for packed
  TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "query must be 3d or 4d.");
  bool is_pack = q.dim() == 3 ? true : false;
  int64_t batch = is_pack ? cu_seq_lens_q.value().size(0) - 1 : q.size(0);
  int qk_head_size = q.size(-1);
  int v_head_size = v.size(-1);
  cnnlQuantizeScheme_t qk_quant_layout = CNNL_QUANTIZE_NONE;
  cnnlQuantizeScheme_t v_quant_layout = CNNL_QUANTIZE_NONE;

  if (has_block_table) {
    TORCH_CHECK(block_tables.value().dim() == 2, "block_tables must be 2d.");
    TORCH_CHECK(k.dim() == 4, "with block table, key_cache must be 4d.");
    TORCH_CHECK(v.dim() == 4, "with block table, value_cache must be 4d.");
    int max_num_blocks_per_seq = block_tables.value().size(1);
    int num_blocks = k.size(0);
    int block_size = k.size(2);
    int k_head_num = k.size(1);
    CHECK_SHAPE(k, num_blocks, k_head_num, block_size, qk_head_size);
    CHECK_SHAPE(v, num_blocks, k_head_num, block_size, v_head_size);
    if (max_num_blocks_per_seq > 1) {  // paged
      CHECK_SHAPE(cu_seq_lens_kv.value(), batch + 1);
    }
  } else {
    // 3d for packed
    TORCH_CHECK(k.dim() == 3 || k.dim() == 4, "key_cache must be 3d or 4d.");
    TORCH_CHECK(v.dim() == 3 || v.dim() == 4, "value_cache must be 3d or 4d.");
    if (k.dim() == 3) {  // packed_kv
      CHECK_SHAPE(cu_seq_lens_kv.value(), batch + 1);
    }
  }

  if (q_quant_scale.has_value()) {
    TORCH_CHECK(k_quant_scale.has_value() &&
                    q_quant_scale.value().dim() == k_quant_scale.value().dim(),
                "q/k must have save quant_layout");
    if (q_quant_scale.value().dim() == 1) {  // fp8
      qk_quant_layout = CNNL_QUANTIZE_PER_TENSOR;
    } else if (q_quant_scale.value().dim() == 3) {
      TORCH_CHECK(!has_block_table, "sage attention not support block_tables.")
      TORCH_CHECK(!v_quant_scale.has_value(),
                  "sage attention need v not-qunat.")
      qk_quant_layout = CNNL_QUANTIZE_PER_BLOCK;
    }
  }
  if (v_quant_scale.has_value()) {
    CHECK_SHAPE(v_quant_scale.value(), 1);
    v_quant_layout = CNNL_QUANTIZE_PER_TENSOR;
  }
  TORCH_CHECK(!out_quant_scale.has_value(),
              "out_quant_scale not support, for reserve");
  // Convert torch tensor to tensor descs
  auto descs = createTensorDescs({q,
                                  k,
                                  v,
                                  cu_seq_lens_q.value_or(at::Tensor()),
                                  cu_seq_lens_kv.value_or(at::Tensor()),
                                  alibi_slope.value_or(at::Tensor()),
                                  attn_bias.value_or(at::Tensor()),
                                  block_tables.value_or(at::Tensor()),
                                  output,
                                  output_lse.value_or(at::Tensor()),
                                  q_quant_scale.value_or(at::Tensor()),
                                  k_quant_scale.value_or(at::Tensor()),
                                  v_quant_scale.value_or(at::Tensor()),
                                  out_quant_scale.value_or(at::Tensor())});

  // Get current handle.
  const torch_mlu::mlu::MLUGuard device_guard(q.device());
  cnnlHandle_t handle = torch_mlu::getCurrentHandle();
  // Get workspace size and malloc workspace.
  cnnlDataType_t cnnl_compute_dtype =
      compute_dtype == "float"  ? CNNL_DTYPE_FLOAT
      : compute_dtype == "half" ? CNNL_DTYPE_HALF
                                : CNNL_DTYPE_BFLOAT16;
  size_t workspace_size = 0;
  CNNL_CHECK_FATAL(
      cnnlGetScaledDotProductAttnWorkspaceSize_v3(handle,
                                                  nullptr /*op_desc*/,
                                                  nullptr /*quant_desc*/,
                                                  descs[0].get(),
                                                  descs[1].get(),
                                                  descs[2].get(),
                                                  descs[3].get(),
                                                  descs[4].get(),
                                                  descs[6].get(),
                                                  descs[5].get(),
                                                  descs[7].get(),
                                                  max_seq_len_q,
                                                  max_seq_len_kv,
                                                  is_causal,
                                                  window_size_left,
                                                  window_size_right,
                                                  return_lse,
                                                  CNNL_COMPUTATION_FAST,
                                                  cnnl_compute_dtype,
                                                  &workspace_size));
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             q.options().dtype(at::kByte));

  // call cnnl extra op.
  CNNL_CHECK_FATAL(cnnlScaledDotProductAttn_v5(handle,
                                               nullptr,
                                               nullptr,
                                               descs[0].get(),
                                               getAtTensorPtr(q),
                                               descs[1].get(),
                                               getAtTensorPtr(k),
                                               descs[2].get(),
                                               getAtTensorPtr(v),
                                               descs[3].get(),
                                               getAtTensorPtr(cu_seq_lens_q),
                                               descs[4].get(),
                                               getAtTensorPtr(cu_seq_lens_kv),
                                               nullptr,
                                               nullptr,
                                               descs[6].get(),
                                               getAtTensorPtr(attn_bias),
                                               descs[5].get(),
                                               getAtTensorPtr(alibi_slope),
                                               descs[7].get(),
                                               getAtTensorPtr(block_tables),
                                               descs[10].get(),
                                               getAtTensorPtr(q_quant_scale),
                                               descs[11].get(),
                                               getAtTensorPtr(k_quant_scale),
                                               descs[12].get(),
                                               getAtTensorPtr(v_quant_scale),
                                               descs[13].get(),
                                               getAtTensorPtr(out_quant_scale),
                                               qk_quant_layout,
                                               qk_quant_layout,
                                               v_quant_layout,
                                               CNNL_QUANTIZE_NONE,
                                               max_seq_len_q,
                                               max_seq_len_kv,
                                               is_causal,
                                               window_size_left,
                                               window_size_right,
                                               softmax_scale,
                                               CNNL_COMPUTATION_FAST,
                                               cnnl_compute_dtype,
                                               getAtTensorPtr(workspace),
                                               workspace_size,
                                               return_lse,
                                               descs[9].get(),
                                               getAtTensorPtr(output_lse),
                                               descs[8].get(),
                                               getAtTensorPtr(output)));
}

}  // namespace xllm::mlu
