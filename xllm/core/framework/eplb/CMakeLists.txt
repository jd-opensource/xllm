include(cc_binary)
include(cc_library)
include(cc_test)

include_directories(
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend/core/include
)

cc_library(
  NAME 
    eplb
  HDRS
    eplb_executor.h
    eplb_manager.h
    eplb_policy.h
    expert_weight_buffer_shm.h
    expert_buffer_manager.h
  SRCS 
    eplb_executor.cpp
    eplb_manager.cpp
    eplb_policy.cpp
    expert_weight_buffer_shm.cpp
    expert_buffer_manager.cpp
  DEPS
    :request
    :common
    glog::glog
    torch
    :platform
)

set(TEST_SRCS
  eplb_policy_test.cpp
)

cc_test(
  NAME
    eplb_policy_test
  SRCS
    ${TEST_SRCS}
  DEPS
    torch
    :eplb
    fmt::fmt
    GTest::gtest_main
)

target_link_libraries(eplb_policy_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)
