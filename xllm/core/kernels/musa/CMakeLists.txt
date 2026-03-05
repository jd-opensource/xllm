include(cc_library)

file(GLOB_RECURSE MUSA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/cuda/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cuh"
)

file(GLOB_RECURSE MUSA_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../cuda/activation.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../cuda/norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../cuda/reshape_paged_cache.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../cuda/rope.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../cuda/*.cpp"
)

cc_library(
  NAME
    musa_kernels
  HDRS
    ${MUSA_HEADER_FILES}
  SRCS
    ${MUSA_SOURCE_FILES}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
    MTTOplib
)
