include(cc_library)
include(cc_test)

if(USE_NPU)
  add_subdirectory(npu)
endif()

cc_library(
  NAME
    platform
  HDRS
    stream.h
    device.h
    vmm_api.h
    shared_vmm_allocator.h
    vmm_torch_allocator.h
  SRCS
    stream.cpp
    device.cpp
    vmm_api.cpp
    shared_vmm_allocator.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_MLU}>:torch_mlu>
    $<$<BOOL:${USE_MLU}>:cnrt>
    $<$<BOOL:${USE_MLU}>:cndrv>
    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_ILU}>>:cuda>
    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_ILU}>>:cudart>
    $<$<BOOL:${USE_MUSA}>:musa>
    $<$<BOOL:${USE_MUSA}>:musart>
    $<$<BOOL:${USE_NPU}>:platform_npu>
)

# TODO: test on multi device
if(USE_NPU OR USE_CUDA OR USE_ILU)
cc_test(
  NAME
    multi_platform_vmm_test
  SRCS
    shared_vmm_allocator_test.cpp
  DEPS
    :platform
    :flags
    GTest::gtest_main
    glog::glog
)
endif()

