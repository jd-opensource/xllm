include(cc_library)

file(GLOB_RECURSE MUSA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.muh"
)

file(GLOB_RECURSE MUSA_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.mu"
)

cc_library(
  NAME
    musa_kernels
  HDRS
    ${MUSA_HEADER_FILES}
  SRCS
    ${MUSA_SOURCE_FILES}
  DEPS
    torch
    util
    MTTOplib
)
