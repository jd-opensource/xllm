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
// #include <tvm/ffi/container/tensor.h>
// #include <tvm/ffi/extra/module.h>
// #include <ATen/DLConvertor.h>

#include <iostream>

#include "function_factory.h"
#include "musa_ops_api.h"

namespace xllm::kernel::musa {

void batch_prefill(torch::Tensor& float_workspace_buffer,
                   torch::Tensor& int_workspace_buffer,
                   torch::Tensor& page_locked_int_workspace_buffer,
                   torch::Tensor& query,
                   torch::Tensor& key,
                   torch::Tensor value,
                   torch::Tensor& q_cu_seq_lens,
                   torch::Tensor& kv_cu_seq_lens,
                   int max_seqlen_q,
                   int max_seqlen_kv,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph) {
  torch::Tensor lse_mate, temp_a, temp_b;
  std::tie(output,
           lse_mate,
           temp_a,
           temp_b) =
      FunctionFactory::get_instance().mate_func().call(
          query,           //"Tensor q,"
          key,             //"Tensor k,"
          value,           //"Tensor v,"
          std::nullopt,    // "Tensor(k_new!)? k_new = None,"
          std::nullopt,    // "Tensor(v_new!)? v_new = None,"
          std::nullopt,    // "Tensor? q_v = None,"
          output,          // "Tensor(out!)? out = None,"
          q_cu_seq_lens,   // "Tensor? cu_seqlens_q = None,"
          kv_cu_seq_lens,  // "Tensor? cu_seqlens_k = None,"
          std::nullopt,    // "Tensor? cu_seqlens_k_new = None,"
          std::nullopt,    // "Tensor? seqused_q = None,"
          std::nullopt,    // "Tensor? seqused_k = None,"
          max_seqlen_q,    // "int? max_seqlen_q = None,"
          max_seqlen_kv,   // "int? max_seqlen_k = None,"
          std::nullopt,    // "Tensor? page_table = None,"
          std::nullopt,    // "Tensor? kv_batch_idx = None,"
          std::nullopt,    // "Tensor? leftpad_k = None,"
          std::nullopt,    // "Tensor? rotary_cos = None,"
          std::nullopt,    // "Tensor? rotary_sin = None,"
          std::nullopt,    // "Tensor? seqlens_rotary = None,"
          std::nullopt,    // "Tensor? q_descale = None,"
          std::nullopt,    // "Tensor? k_descale = None,"
          std::nullopt,    // "Tensor? v_descale = None,"
          std::nullopt,    // "float? softmax_scale = None,"
          true,            // "bool is_causal = False,"
          -1,              // "int window_size_left = -1,"
          -1,              // "int window_size_right = -1,"
          0,               // "int attention_chunk = 0,"
          0.f,             // "float softcap = 0.0,"
          false,           // "bool is_rotary_interleaved = False,"
          std::nullopt,    // "Tensor? scheduler_metadata = None,"
          1,               // "int num_splits = 1,"
          std::nullopt,    // "bool? pack_gqa = None,"
          0                // "int sm_margin = 0
      );
}

}  // namespace xllm::kernel::musa
