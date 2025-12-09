include(cc_library)
set(CMAKE_CUDA_ARCHITECTURES ivcore11)
file(GLOB_RECURSE ILU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE ILU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

cc_library(
  NAME
    ilu_kernels
  HDRS
    ${ILU_HEADER_FILES}
  SRCS
    ${ILU_SOURCE_FILES}
  DEPS
    torch
    :util
    ixformer_kernels
    ixformer
    ${Python3_LIBRARIES}
    cuinfer
)
