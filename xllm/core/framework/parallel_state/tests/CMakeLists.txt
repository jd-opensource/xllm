include(cc_test)

if(NOT USE_NPU)
  # Add test for reduce_scatter
  # This test must exist individually, because it contains forked processes
  # which does not allow any device init on main process
  cc_test(
    NAME
      parallel_state_test
    SRCS
      parallel_state_tests.cpp
    DEPS
      :parallel_state
      GTest::gtest_main
      torch
      glog::glog
    )
endif()
