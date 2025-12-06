include(cc_library)

add_subdirectory(xllm_ops)

file(GLOB_RECURSE XLLM_CORE_KERNELS_NPU_HEADER
  "${CMAKE_CURRENT_LIST_DIR}/custom_functions_npu/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/ops_npu/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE XLLM_CORE_KERNELS_NPU_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/custom_functions_npu/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ops_npu/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

cc_library(
  NAME
    npu_kernels
  HDRS
    ${XLLM_CORE_KERNELS_NPU_HEADER}
  SRCS
    ${XLLM_CORE_KERNELS_NPU_SRCS}
  DEPS
    :model_context
    glog::glog
    torch
    torch_npu
)
