include(cc_test)

if(USE_NPU OR USE_CUDA OR USE_ILU)
  cc_test(
    NAME
      multi_platform_vmm_test
    SRCS
      shared_vmm_allocator_test.cpp
    DEPS
      :platform
      :flags
      GTest::gtest_main
      glog::glog
  )
endif()
