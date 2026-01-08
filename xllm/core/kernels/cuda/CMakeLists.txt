include(cc_library)
include(cc_test)

#
# Fetch CUTLASS dependency
#
include(FetchContent)
set(CUTLASS_VERSION "v4.2.1")
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://gh-proxy.com/https://github.com/nvidia/cutlass.git
  GIT_TAG        ${CUTLASS_VERSION}
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cutlass)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})

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

add_subdirectory(tests)
