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

# Define the library
cc_library(
  NAME 
    models
  HDRS
    model_registry.h
    models.h
    deepseek_v2.h
    deepseek_v2_mtp.h
    llama.h
    minicpmv.h
    qwen2.h
    qwen2_5_vl.h
    qwen3.h
    qwen3_moe.h
    qwen_base.h
  SRCS
    model_registry.cpp
  DEPS
    :model
)
