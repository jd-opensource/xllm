include(cc_test)

cc_test(
  NAME
    decoder_reshape_and_cache_test
  SRCS
    decoder_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(decoder_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    prefill_reshape_and_cache_test
  SRCS
    prefill_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(prefill_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    cache_select_test
  SRCS
    cache_select_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cache_select_test PRIVATE brpc)
