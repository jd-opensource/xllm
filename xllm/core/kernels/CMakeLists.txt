include(cc_library)

if(USE_NPU)
  add_subdirectory(npu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()


cc_library(
  NAME
    kernels
  HDRS
    param.h
    ops_api.h
  SRCS
    ops_api.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:npu_kernels>
    $<$<BOOL:${USE_MLU}>:mlu_kernels>
)