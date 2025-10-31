include(cc_library)

include_directories(
  ${CMAKE_SOURCE_DIR}/third_party/dlpack/include
)

include_directories(
  ${CMAKE_SOURCE_DIR}/third_party/tvm-ffi/include
)

if(USE_NPU)
  add_subdirectory(npu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()

if(USE_CUDA)
  add_subdirectory(cuda)
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
    $<$<BOOL:${USE_CUDA}>:cuda_kernels>
)