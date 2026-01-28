include(cc_library)
include(cc_test)

# Define the base dependencies
set(BASE_DEPS
  :common
  :kv_cache
  :prefix_cache
  :block
  :chat_template
  glog::glog
  torch
  torch_python
)


# Define the library
cc_library(
  NAME 
    processors
  HDRS
    image_processor.h
    clip_image_processor.h
    minicpmv_image_processor.h
    qwen2_vl_image_processor.h
    glm4v_image_processor.h
    pywarpper_image_processor.h
    input_processor.h
    siglip_image_processor.h
  SRCS
    image_processor.cpp
    clip_image_processor.cpp
    minicpmv_image_processor.cpp
    qwen2_vl_image_processor.cpp
    glm4v_image_processor.cpp
    pywarpper_image_processor.cpp
    siglip_image_processor.cpp
  DEPS
    ${BASE_DEPS}
)
