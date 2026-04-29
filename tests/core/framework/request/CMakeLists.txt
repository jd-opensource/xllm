include(cc_test)

cc_test(
  NAME
    sample_slot_test
  SRCS
    sample_slot_test.cpp
  DEPS
    :request
    GTest::gtest
    GTest::gtest_main
)
