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
  SRCS
    model_registry.cpp
  DEPS
    :model
)
