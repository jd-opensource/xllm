include(cc_library)

cc_library(
  NAME 
    platform
  HDRS
    stream_helper.h
  SRCS
    stream_helper.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
)