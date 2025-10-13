include(cc_library)

cc_library(
  NAME 
    platform
  HDRS
    stream_helper.h
    device.h
  SRCS
    stream_helper.cpp
    device.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:ascendcl>
)