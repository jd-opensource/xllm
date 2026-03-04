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

#pragma once

#include <unordered_set>

#include "models/model_registry.h"

namespace xllm {

REGISTER_MODEL_ARGS(onerec, [&] {
  LOAD_ARG_OR(model_type, "model_type", "onerec");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");

  LOAD_ARG_OR(hidden_size, "d_model", 128);
  LOAD_ARG_OR(intermediate_size, "d_ff", 256);

  LOAD_ARG_OR(n_layers, "num_decoder_layers", 4);
  LOAD_ARG_OR(n_encoder_layers, "num_layers", 12);

  LOAD_ARG_OR(n_heads, "num_heads", 4);
  LOAD_ARG_OR(head_dim, "d_kv", 32);
  LOAD_ARG_OR_FUNC(
      decoder_n_heads, "decoder_num_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      decoder_head_dim, "decoder_d_kv", [&] { return args->head_dim(); });

  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG(decoder_n_kv_heads, "decoder_num_key_value_heads");

  LOAD_ARG_OR(vocab_size, "vocab_size", 8200);
  LOAD_ARG_OR(rms_norm_eps, "layer_norm_epsilon", 1e-6);
  LOAD_ARG_OR(max_position_embeddings, "max_length", 500);
  LOAD_ARG_OR(use_absolute_position_embedding,
              "use_absolute_position_embedding",
              false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", true);

  LOAD_ARG_OR(use_moe, "use_moe", false);
  LOAD_ARG_OR(moe_score_func, "moe_score_func", "softmax");
  LOAD_ARG_OR(moe_route_scale, "moe_route_scale", 1.0f);
  LOAD_ARG_OR(n_routed_experts, "moe_num_experts", 8);
  LOAD_ARG_OR(moe_use_shared_experts, "moe_use_shared_experts", false);
  LOAD_ARG_OR(n_shared_experts, "moe_num_shared_experts", 0);
  LOAD_ARG_OR(num_experts_per_tok, "moe_topk", 2);
  LOAD_ARG_OR(moe_intermediate_size, "moe_inter_dim", 1024);

  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

REGISTER_TOKENIZER_ARGS(onerec, [&] { SET_ARG(tokenizer_type, "rec"); });

}  // namespace xllm
