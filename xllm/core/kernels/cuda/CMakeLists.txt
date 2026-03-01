include(cc_library)
include(cc_test)

file(GLOB_RECURSE CUDA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cuh"
)

file(GLOB_RECURSE CUDA_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${CUDA_HEADER_FILES}
  SRCS
    ${CUDA_SOURCE_FILES}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
)

cc_test(
  NAME
    decoder_reshape_and_cache_test
  SRCS
    xattention/decoder_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(decoder_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    prefill_reshape_and_cache_test
  SRCS
    xattention/prefill_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(prefill_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    cache_select_test
  SRCS
    xattention/cache_select_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cache_select_test PRIVATE brpc)

cc_test(
  NAME
    moe_topk_funcs_test
  SRCS
    moe_topk_funcs_test.cu
  DEPS
    :cuda_kernels
    GTest::gtest_main
)