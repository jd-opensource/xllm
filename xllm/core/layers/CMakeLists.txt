include(cc_library)

add_subdirectory(common)
if(USE_NPU)
  add_subdirectory(npu)
  add_subdirectory(npu_torch)
elseif(USE_MLU)
  add_subdirectory(mlu)
elseif(USE_ILU)
  add_subdirectory(ilu)
else()
  add_subdirectory(cuda)
endif()

cc_library(
  NAME
    attention_mask
  HDRS
    common/attention_mask.h
  SRCS
    common/attention_mask.cpp
  DEPS
    :state_dict
    :block
    :kv_cache
    :prefix_cache
    glog::glog
    gflags::gflags
    torch
)


set(LAYERS_HDRS
    common/multi_head_attention.h
)

if(NOT USE_NPU)
    list(APPEND LAYERS_HDRS
      config.h
      deepseek_v2_decoder_layer.h
      deepseek_v32_decoder_layer.h
      glm4_vision_encode_layer.h
      llama_decoder_layer.h
      qwen2_decoder_layer.h
      qwen2_vision_encode_layer.h
      qwen2dot5_vision_encode_layer.h
      qwen3_vision_encode_layer.h
      qwen3_decoder_layer.h
      qwen3_moe_decoder_layer.h
      glm4_decoder_layer.h
      siglip_encoder_layer.h
      pos_embedding.h
      word_embedding.h
      lm_head.h
    )
endif()

cc_library(
  NAME
    layers
  HDRS
    ${LAYERS_HDRS}
  SRCS
    common/multi_head_attention.cpp
  DEPS
    $<$<BOOL:${USE_NPU}>:npu_layers>
    $<$<BOOL:${USE_NPU}>:npu_torch_layers>
    $<$<BOOL:${USE_MLU}>:mlu_layers>
    $<$<BOOL:${USE_CUDA}>:cuda_layers>
    $<$<BOOL:${USE_ILU}>:ilu_layers>
    :parallel_state
    :state_dict
    :kv_cache
    :prefix_cache
    :block
    :attention_mask
    glog::glog
    gflags::gflags
    torch
)
