include(cc_library)
include(cc_test)

# CUTLASS headers are provided by root CMakeLists.txt via submodule (third_party/cutlass)
include_directories(${CMAKE_CURRENT_LIST_DIR})

#
# CUDA feature detection
#
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  # SM90+ (Hopper/Blackwell) scaled matmul support
  add_compile_definitions(
    ENABLE_SCALED_MM_SM90=1
    ENABLE_SCALED_MM_SM100=1
    ENABLE_SCALED_MM_SM120=1
  )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  add_compile_definitions(ENABLE_FP8)
endif()

file(GLOB_RECURSE CUDA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cuh"
)

file(GLOB_RECURSE CUDA_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)

# Exclude test files from source list (tests are built separately)
list(FILTER CUDA_SOURCE_FILES EXCLUDE REGEX ".*/tests/.*")

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
    glog::glog
)
add_subdirectory(tests)
