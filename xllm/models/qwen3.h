#pragma once

#include "core/layers/npu/qwen3_decoder_layer.h"
#include "qwen_base.h"

namespace xllm::hf {

class QWen3DecoderLayerImpl : public QWenDecoderLayerImplBase<Qwen3Decoder> {
 public:
  QWen3DecoderLayerImpl(const Context& context)
      : QWenDecoderLayerImplBase<Qwen3Decoder>(context) {}
};
TORCH_MODULE(QWen3DecoderLayer);

torch::Tensor get_qwen3_rotary_embedding(int64_t dim,
                                         int64_t seq_len,
                                         double rope_theta,
                                         const torch::TensorOptions& options) {
  return get_qwen_concat_rotary_embedding(dim, seq_len, rope_theta, options);
}

class QWen3ModelImpl : public QWenModelImplBase<QWen3DecoderLayer> {
 public:
  QWen3ModelImpl(const Context& context)
      : QWenModelImplBase<QWen3DecoderLayer>("qwen3",
                                             context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
#if defined(USE_NPU)
    work_space_ = AtbWorkspace(options.device());
    embed_tokens_ = register_module("embed_tokens", AtbWordEmbedding(context));
    norm_ = register_module("norm", RmsNorm(context));

    atb_pos_emb_ = AtbRotaryEmbedding(context);
    cos_sin_ = get_qwen3_rotary_embedding(128,
                                          model_args.max_position_embeddings(),
                                          model_args.rope_theta(),
                                          options);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    // encode_attn_mask_ =
    //   AttentionMaskImpl(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
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
#elif defined(USE_MLU)
    norm_ = register_module("norm", RMSNormResidual(context));
    embed_tokens_ = register_module("embed_tokens", ParallelEmbedding(context));
#endif

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public QWenForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const Context& context)
      : QWenForCausalLMImplBase<QWen3Model>(context) {}
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
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

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
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
