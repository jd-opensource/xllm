include(cc_library)
include(cc_test)

# Define the base dependencies
set(BASE_DEPS
  :common
  :flags
  :layers
  :kv_cache
  :prefix_cache
  :block
  :processors
  :chat_template
  glog::glog
  torch
)

if(USE_NPU)
  # Modify dependencies for NPU
  list(APPEND BASE_DEPS torch_npu)
  list(APPEND BASE_DEPS :npu_layers)
endif()

if(USE_MLU)
  # Modify dependencies for MLU
  list(APPEND BASE_DEPS :mlu_layers)
endif()

# Define the library
cc_library(
  NAME 
    models
  HDRS
    model_registry.h
    models.h
    $<$<BOOL:${USE_NPU}>:deepseek_v2.h>
    $<$<BOOL:${USE_NPU}>:deepseek_v2_mtp.h>
    $<$<BOOL:${USE_NPU}>:llama.h>
    $<$<BOOL:${USE_NPU}>:minicpmv.h>
    $<$<BOOL:${USE_NPU}>:qwen2.h>
    $<$<BOOL:${USE_NPU}>:qwen2_5_vl.h>
    $<$<BOOL:${USE_NPU}>:qwen3_moe.h>
    qwen3.h
    qwen_base.h
  SRCS
    model_registry.cpp
  DEPS
    :model
    ${BASE_DEPS}
)
