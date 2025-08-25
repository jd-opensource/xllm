include(cc_library)

file(GLOB_RECURSE HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.hpp"
)

file(GLOB_RECURSE SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

cc_library(
  NAME
    xllm_ops
  HDRS
    ${HEADER_FILES}
  SRCS
    ${SOURCE_FILES}
  DEPS
    atb
    $<$<BOOL:${USE_NPU}>:torch_npu>
    gflags::gflags
    nlohmann_json::nlohmann_json
    opapi
    spdlog::spdlog
)
