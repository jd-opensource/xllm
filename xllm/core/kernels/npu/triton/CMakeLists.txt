include(cc_library)
include(cc_binary)
message("===========get in build triton_ascend_loader==============")

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(PLATFORM "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(PLATFORM "aarch64-linux")
endif()

set(TORCH_API_HDRS
  csrc/torch_api/npu_triton_fused_gdn_gating.h
)

set(TORCH_API_SRCS
  csrc/torch_api/npu_triton_fused_gdn_gating.cpp
)

cc_library(
  NAME
    triton_adapter
  HDRS
    include/kernel_handle.h
    include/kernel_registry.h
    include/kernel_loader.h
    include/macros.h
    include/kernel_launchers.h
    npu_triton_add_kernel.h
    ${TORCH_API_HDRS}
  INCLUDES
    ${ASCEND_HOME_PATH}/${PLATFORM}/include/experiment/platform
    ${ASCEND_HOME_PATH}/${PLATFORM}/include/experiment/msprof
  LINKOPTS
    "${ASCEND_HOME_PATH}/lib64/libruntime.so"
  SRCS
    csrc/kernel_handle.cpp
    csrc/kernel_registry.cpp
    csrc/kernel_loader.cpp
    npu_triton_add_kernel.cpp
    ${TORCH_API_SRCS}
  DEPS
    ascendcl
    nlohmann_json::nlohmann_json
    # spdlog::spdlog
)


#cc_binary(
#  NAME
#    test_add_kernel
#  SRCS
#    test/test_triton_add_kernel.cpp
#  DEPS
#    triton_adapter
#    torch
#    torch_npu
#)

add_subdirectory(test)
