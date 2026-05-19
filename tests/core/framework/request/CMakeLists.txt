include(cc_test)

cc_test(
  NAME
    sample_slot_test
  SRCS
    sample_slot_test.cpp
  DEPS
    :config
    :request
    GTest::gtest
    GTest::gtest_main
)

cc_test(
  NAME
    request_params_test
  SRCS
    request_params_test.cpp
  DEPS
    :request
    GTest::gtest_main
)

cc_test(
  NAME
    sequence_kv_state_test
  SRCS
    sequence_kv_state_test.cpp
  DEPS
    :request
    GTest::gtest_main
)
