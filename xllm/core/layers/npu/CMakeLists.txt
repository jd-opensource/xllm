include(cc_library)
include(cc_test)


cc_library(
  NAME
    npu_layers 
  HDRS
    atb_base.h
    word_embedding.h
    atb_word_embedding_impl.h
    pos_embedding.h
    llm_head.h
    atb_head_impl.h
    $<$<BOOL:${USE_A2}>:npu_qwen2dot5_vision_encoder_layer_impl.h>
    $<$<BOOL:${USE_A2}>:npu_qwen3_moe_decoder_layer_impl.h>
    # atb_parallel_linear.h
    buffer/atb_buffer.h
    buffer/atb_workspace.h
    npu_base_layer.h
    npu_column_parallel_linear_impl.h
    npu_deepseek_v2_decoder_layer_impl.h
    npu_llama_decoder_layer_impl.h
    npu_qwen2_decoder_layer_impl.h
    npu_qwen3_decoder_layer_impl.h
    npu_rms_norm_impl.h
    npu_siglip_encoder_layer_impl.h
  SRCS
    $<$<BOOL:${USE_A2}>:npu_qwen2dot5_vision_encoder_layer_impl.cpp>
    $<$<BOOL:${USE_A2}>:npu_qwen3_moe_decoder_layer_impl.cpp>
    atb_base.cpp
    word_embedding.cpp
    atb_word_embedding_impl.cpp
    pos_embedding.cpp
    llm_head.cpp
    atb_head_impl.cpp
    # atb_parallel_linear.cpp
    buffer/atb_buffer.cpp
    buffer/atb_workspace.cpp
    npu_base_layer.cpp
    npu_column_parallel_linear_impl.cpp
    npu_deepseek_v2_decoder_layer_impl.cpp
    npu_llama_decoder_layer_impl.cpp
    npu_qwen2_decoder_layer_impl.cpp
    npu_qwen3_decoder_layer_impl.cpp
    npu_rms_norm_impl.cpp
    npu_siglip_encoder_layer_impl.cpp
  DEPS
    "-Wl,--whole-archive"
    "-Wl,--no-whole-archive"
    :base_layer
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




