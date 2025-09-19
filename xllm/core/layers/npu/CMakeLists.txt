include(cc_library)
include(cc_test)


cc_library(
  NAME
    npu_layers 
  HDRS
    npu_word_embedding_impl.h
    npu_pos_embedding_impl.h
    npu_lm_head_impl.h
    npu_qwen2dot5_vision_encoder_layer_impl.h
    npu_qwen3_moe_decoder_layer_impl.h
    # atb_parallel_linear.h
    npu_block_copy_impl.h
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
    npu_word_embedding_impl.cpp
    npu_pos_embedding_impl.cpp
    npu_lm_head_impl.cpp
    npu_qwen2dot5_vision_encoder_layer_impl.cpp
    npu_qwen3_moe_decoder_layer_impl.cpp
    # atb_parallel_linear.cpp
    npu_block_copy_impl.cpp
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
    :eplb
    :parallel_state
    :state_dict
    glog::glog
    gflags::gflags
    spdlog::spdlog
    torch
    torch_npu
    xllm_kernels
)

cc_test(
  NAME
    rms_norm_test
  SRCS
    rms_norm_test.cpp
  DEPS
    :npu_layers
    :model_context
    GTest::gtest
    GTest::gtest_main
    torch
    torch_npu
    xllm_kernels
    opapi
)

target_link_libraries(rms_norm_test
                 PUBLIC
                 $<$<BOOL:${USE_NPU}>:ascendcl>
                 $<$<BOOL:${USE_NPU}>:hccl>
                 $<$<BOOL:${USE_NPU}>:c_sec>
                 $<$<BOOL:${USE_NPU}>:atb>
                 $<$<BOOL:${USE_NPU}>:nnopbase>)


