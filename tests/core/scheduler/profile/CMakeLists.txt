include(cc_test)

cc_test(
  NAME
    profile_graph_warmup_test
  SRCS
    profile_graph_warmup_test.cpp
  DEPS
    :block
    :profile
    :request
    GTest::gtest_main
)
