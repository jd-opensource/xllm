include(cc_test)

get_filename_component(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# 构造 binary 目录的绝对路径
set(BINARY_DIR "${TEST_DIR}/binary")

cc_test(
  NAME
    fused_gdn_gating_test
  SRCS
    triton_fused_gdn_gating_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

target_compile_definitions(fused_gdn_gating_test PRIVATE
    TEST_BINARY_DIR="${BINARY_DIR}"
)
