include(cc_test)

cc_test(
  NAME
    fast_tokenizer_test
  SRCS
    fast_tokenizer_test.cpp
  DEPS
    :tokenizer
    glog::glog
    GTest::gtest_main
)
