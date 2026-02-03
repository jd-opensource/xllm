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

#include "qwen3.h"

namespace xllm {
// register the causal model
REGISTER_CAUSAL_MODEL(llama, QWen3ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(llama, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llama");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  // decide model type based on vocab size
  LOAD_ARG_OR(vocab_size, "vocab_size", 128256);
  if (args->vocab_size() == 128256) {
    // choose the right chat template
    SET_ARG(model_type, "llama3");

    LOAD_ARG_OR(hidden_size, "hidden_size", 8192);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 28672);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 128000);
    // TODO: support a list of eos token ids
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
    LOAD_ARG_OR(rope_theta, "rope_theta", 500000.0f);
    // load rope scaling parameters
    LOAD_ARG(rope_scaling_rope_type, "rope_scaling.rope_type");
    LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
    LOAD_ARG(rope_scaling_low_freq_factor, "rope_scaling.low_freq_factor");
    LOAD_ARG(rope_scaling_high_freq_factor, "rope_scaling.high_freq_factor");
    LOAD_ARG(rope_scaling_original_max_position_embeddings,
             "rope_scaling.original_max_position_embeddings");
    // stop token ids: "<|eom_id|>", "<|eot_id|>"
    SET_ARG(stop_token_ids, std::unordered_set<int32_t>({128008, 128009}));
  } else {
    // llama 2
    LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
    LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
    // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  }

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});
}  // namespace xllm