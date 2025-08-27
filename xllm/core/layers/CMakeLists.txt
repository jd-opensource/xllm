include(cc_library)
include(cc_test)


cc_library(
  NAME 
    rotary_embedding
  HDRS
    rotary_embedding.h
  SRCS
    rotary_embedding.cpp
  DEPS
    :state_dict
    :block
    :kv_cache
    :prefix_cache
    glog::glog
    gflags::gflags
    torch
)

cc_library(
  NAME 
    layers
  DEPS
    :state_dict
    :kv_cache
    :prefix_cache
    :block
    :rotary_embedding
    glog::glog
    gflags::gflags
    torch
)

if(USE_NPU)
  add_subdirectory(npu)
endif()
