include(cc_library)
include(cc_test)


cc_library(
  NAME
    npu_layers 
  HDRS
    llama_decoder_layer.h
    qwen2_decoder_layer.h
    qwen3_decoder_layer.h
    qwen3_moe_decoder_layer.h
    attn_mask.h
    atb_base.h
    word_embedding.h
    atb_word_embedding_impl.h
    pos_embedding.h
    rms_norm.h
    llm_head.h
    atb_head_impl.h
    siglip_encoder_layer.h
    multi_head_attention.h
    $<$<BOOL:${USE_A2}>:qwen2_5_vision_encoder_layer.h>
    $<$<BOOL:${USE_A2}>:qwen3_moe_decoder_layer.h>
    atb_linear.h
    atb_parallel_linear.h
    buffer/atb_buffer.h
    buffer/atb_workspace.h
  SRCS
    llama_decoder_layer.cpp
    qwen2_decoder_layer.cpp
    qwen3_decoder_layer.cpp
    $<$<BOOL:${USE_A2}>:qwen2_5_vision_encoder_layer.cpp>
    $<$<BOOL:${USE_A2}>:qwen3_moe_decoder_layer.cpp>
    attn_mask.cpp
    atb_base.cpp
    word_embedding.cpp
    atb_word_embedding_impl.cpp
    pos_embedding.cpp
    rms_norm.cpp
    llm_head.cpp
    atb_head_impl.cpp
    siglip_encoder_layer.cpp
    multi_head_attention.cpp
    deepseek_v2_decoder_layer.cpp
    atb_linear.cpp
    atb_parallel_linear.cpp
    buffer/atb_buffer.cpp
    buffer/atb_workspace.cpp
  DEPS
    "-Wl,--whole-archive"
    "-Wl,--no-whole-archive"
    :kv_cache
    :prefix_cache
    :block
    :parallel_state
    :state_dict
    glog::glog
    gflags::gflags
    spdlog::spdlog
    torch
    torch_npu
    xllm_kernels
)




