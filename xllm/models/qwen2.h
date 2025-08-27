/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#if defined(USE_NPU)
#include "core/layers/npu/qwen2_decoder_layer.h"
#endif
#include "qwen_base.h"

// QWen2 model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/qwen2/modeling_qwen2.py
namespace xllm::hf {

class QWen2DecoderLayerImpl : public QWenDecoderLayerImplBase<Qwen2Decoder> {
 public:
  QWen2DecoderLayerImpl(const Context& context)
      : QWenDecoderLayerImplBase<Qwen2Decoder>(context) {}
};
TORCH_MODULE(QWen2DecoderLayer);

torch::Tensor get_qwen2_rotary_embedding(int64_t dim,
                                         int64_t seq_len,
                                         double rope_theta,
                                         const torch::TensorOptions& options) {
  return get_qwen_concat_rotary_embedding(dim, seq_len, rope_theta, options);
}

class QWen2ModelImpl : public QWenModelImplBase<QWen2DecoderLayer> {
 public:
  QWen2ModelImpl(const Context& context)
      : QWenModelImplBase<QWen2DecoderLayer>("qwen2",
                                             context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    work_space_ = AtbWorkspace(options.device());
    embed_tokens_ = register_module("embed_tokens", AtbWordEmbedding(context));
    norm_ = register_module("norm", RmsNorm(context));

    atb_pos_emb_ = AtbRotaryEmbedding(context);
    cos_sin_ = get_qwen2_rotary_embedding(
        model_args.hidden_size() / model_args.n_heads(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = AttentionMaskImpl(options.device(),
                                   options.dtype().toScalarType(),
                                   /*mask_value=*/mask_value);
    atb::Status st = atb::CreateContext(&context_);
    LOG_IF(ERROR, st != 0) << "ContextFactory create atb::Context fail";
    device_id = options.device().index();
    void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
    LOG_IF(ERROR, stream == nullptr) << "get current stream fail";
    // context_->SetExecuteStream(atb_speed::Utils::GetCurrentStream());
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen2DecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }
};
TORCH_MODULE(QWen2Model);

class QWen2ForCausalLMImpl : public QWenForCausalLMImplBase<QWen2Model> {
 public:
  QWen2ForCausalLMImpl(const Context& context)
      : QWenForCausalLMImplBase<QWen2Model>(context) {}
};
TORCH_MODULE(QWen2ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(qwen2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For Qwen2/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::hf
