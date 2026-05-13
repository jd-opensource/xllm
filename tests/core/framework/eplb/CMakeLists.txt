include(cc_test)

cc_test(
  NAME
    eplb_policy_test
  SRCS
    eplb_policy_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_policy_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)
