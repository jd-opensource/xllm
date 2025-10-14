include(cc_library)

cc_library(
  NAME 
    platform
  HDRS
    stream.h
    device.h
  SRCS
    stream.cpp
    device.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:ascendcl>
)

if(USE_NPU)
  add_subdirectory(npu)
endif()