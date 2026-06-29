include(cc_library)

set(SKIP_MLU_TRITON_AOT OFF)
if(DEFINED ENV{XLLM_SKIP_MLU_TRITON_AOT})
  string(TOUPPER "$ENV{XLLM_SKIP_MLU_TRITON_AOT}" XLLM_SKIP_MLU_TRITON_AOT)
  if(NOT XLLM_SKIP_MLU_TRITON_AOT STREQUAL ""
     AND NOT XLLM_SKIP_MLU_TRITON_AOT STREQUAL "0"
     AND NOT XLLM_SKIP_MLU_TRITON_AOT STREQUAL "FALSE"
     AND NOT XLLM_SKIP_MLU_TRITON_AOT STREQUAL "OFF")
    set(SKIP_MLU_TRITON_AOT ON)
  endif()
endif()

find_library(MLU_CNRT_LIBRARY
  NAMES cnrt
  PATHS "$ENV{NEUWARE_HOME}/lib64"
)
if(NOT MLU_CNRT_LIBRARY)
  message(FATAL_ERROR "Failed to find MLU libcnrt.so in $ENV{NEUWARE_HOME}/lib64")
endif()

if(SKIP_MLU_TRITON_AOT)
  message(STATUS "Skipping MLU Triton AOT compilation because XLLM_SKIP_MLU_TRITON_AOT is set")
else()
  add_subdirectory(tmo_kernel)
endif()

file(GLOB_RECURSE MLU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE MLU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

set(MLU_TRITON_AOT_STUB_SOURCE
  "${CMAKE_CURRENT_LIST_DIR}/triton_aot_stubs.cpp"
)
set(MLU_TRITON_AOT_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_fn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_update_decode.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/chunk_gated_delta_rule.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fused_gdn_gating.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fused_recurrent_gated_delta_rule.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fused_recurrent_gated_delta_rule_packed_decode.cpp"
)
list(REMOVE_ITEM MLU_SOURCE_FILES ${MLU_TRITON_AOT_STUB_SOURCE})

set(MLU_KERNEL_DEPS
  torch
  :platform
  cnclep
  torch_mlu
  -Wl,--no-as-needed
  torch_mlu_ops
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
  Python::Python
)

if(SKIP_MLU_TRITON_AOT)
  list(REMOVE_ITEM MLU_SOURCE_FILES ${MLU_TRITON_AOT_SOURCE_FILES})
  list(APPEND MLU_SOURCE_FILES ${MLU_TRITON_AOT_STUB_SOURCE})
else()
  list(APPEND MLU_KERNEL_DEPS :tmo_kernels)
endif()

cc_library(
  NAME
    mlu_kernels
  HDRS
    ${MLU_HEADER_FILES}
  SRCS
    ${MLU_SOURCE_FILES}
  DEPS
    ${MLU_KERNEL_DEPS}
)

if(NOT SKIP_MLU_TRITON_AOT)
  target_include_directories(mlu_kernels PRIVATE ${TRITON_KERNEL_INCLUDE_DIR})
  target_link_libraries(mlu_kernels PRIVATE
    ${TRITON_KERNEL_OBJECT_FILES}
    -Wl,--no-as-needed
    ${MLU_CNRT_LIBRARY}
    -Wl,--as-needed
  )
  add_dependencies(mlu_kernels triton_kernel_compile)
endif()
