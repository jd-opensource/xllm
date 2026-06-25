include(cc_library)

find_library(MLU_CNRT_LIBRARY
  NAMES cnrt
  PATHS "$ENV{NEUWARE_HOME}/lib64"
)
if(NOT MLU_CNRT_LIBRARY)
  message(FATAL_ERROR "Failed to find MLU libcnrt.so in $ENV{NEUWARE_HOME}/lib64")
endif()

add_subdirectory(tmo_kernel)

file(GLOB_RECURSE MLU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE MLU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

cc_library(
  NAME
    mlu_kernels
  HDRS
    ${MLU_HEADER_FILES}
  SRCS
    ${MLU_SOURCE_FILES}
  DEPS
    torch
    :platform
    :tmo_kernels
    cnclep
    torch_mlu
    -Wl,--no-as-needed
    torch_mlu_ops
    ${MLU_CNRT_LIBRARY}
    -Wl,--as-needed
    Python::Python
)

target_include_directories(mlu_kernels PRIVATE ${TRITON_KERNEL_INCLUDE_DIR})
target_link_libraries(mlu_kernels PRIVATE
  ${TRITON_KERNEL_OBJECT_FILES}
  -Wl,--no-as-needed
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
)
add_dependencies(mlu_kernels triton_kernel_compile)
