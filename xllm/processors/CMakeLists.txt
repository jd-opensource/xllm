include(cc_library)
include(cc_test)

# Define the base dependencies
set(BASE_DEPS
  :common
  :layers
  :kv_cache
  :prefix_cache
  :block
  :chat_template
  glog::glog
  torch
)

if(USE_NPU)
  # Check if NPU is being used
  include_directories($ENV{XLLM_KERNELS_PATH}/include/xllm_kernels/core/include) 
  include_directories($ENV{XLLM_KERNELS_PATH}/include/xllm_kernels)

  # Modify dependencies for NPU
  list(APPEND BASE_DEPS torch_npu)
  list(APPEND BASE_DEPS :npu_layers)
endif()

# Define the library
cc_library(
  NAME 
    processors
  HDRS
    image_processor.h
    clip_image_processor.h
    minicpmv_image_processor.h
    qwen2_vl_image_processor.h
    qwen3_vl_image_processor.h
    pywarpper_image_processor.h
    input_processor.h
  SRCS
    image_processor.cpp
    clip_image_processor.cpp
    minicpmv_image_processor.cpp
    qwen2_vl_image_processor.cpp
    qwen3_vl_image_processor.cpp
    pywarpper_image_processor.cpp
  DEPS
    ${BASE_DEPS}
)