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
    multimodal_input_processor.h
    multimodal_processor.h
    clip_input_processor.h
    clip_prompt_processor.h
    glm4v_prompt_processor.h
    minicpmv_input_processor.h
    minicpmv_prompt_processor.h
    qwen2_vl_input_processor.h
    qwen2_5_vl_prompt_processor.h
    glm4v_input_processor.h
    pywarpper_input_processor.h
    prompt_processor.h
  SRCS
    multimodal_input_processor.cpp
    clip_input_processor.cpp
    clip_prompt_processor.cpp
    glm4v_prompt_processor.cpp
    minicpmv_input_processor.cpp
    minicpmv_prompt_processor.cpp
    qwen2_vl_input_processor.cpp
    qwen2_5_vl_prompt_processor.cpp
    glm4v_input_processor.cpp
    pywarpper_input_processor.cpp
  DEPS
    ${BASE_DEPS}
)
