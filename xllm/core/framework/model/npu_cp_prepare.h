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

#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace xllm {

struct CpPrefillInputs {
  torch::Tensor cp_load_balance_idx;
  torch::Tensor cp_o_recover_idx;
  torch::Tensor cp_kv_recover_idx;

  torch::Tensor k_gather_index_prev;
  torch::Tensor k_gather_index_next;

  torch::Tensor actual_seq_lengths_query_prev;
  torch::Tensor actual_seq_lengths_query_next;
  torch::Tensor actual_seq_lengths_key_prev;
  torch::Tensor actual_seq_lengths_key_next;
};

CpPrefillInputs prepare_cp_prefill_inputs(int cp_size,
            const torch::Tensor& input_ids,
            const torch::Tensor& position_ids,
            const torch::Tensor& input_lengths);

}  // namespace xllm
