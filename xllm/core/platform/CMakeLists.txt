include(cc_library)

cc_library(
  NAME
    platform
  HDRS
    stream.h
    device.h
    vmm_api.h
  SRCS
    stream.cpp
    device.cpp
    vmm_api.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_MLU}>:torch_mlu>
    $<$<BOOL:${USE_MLU}>:cnrt>
    $<$<BOOL:${USE_MLU}>:cndrv>
    $<$<BOOL:${USE_CUDA}>:cuda>
    $<$<BOOL:${USE_CUDA}>:cudart>
)

if(USE_NPU)
  add_subdirectory(npu)
endif()
