include(cc_test)

if(USE_NPU OR USE_CUDA OR USE_ILU OR USE_DCU)
  cc_test(
    NAME
      multi_platform_vmm_test
    SRCS
      shared_vmm_allocator_test.cpp
    DEPS
      :platform
      :config
      GTest::gtest_main
      glog::glog
  )
  if(USE_DCU)
    target_compile_definitions(multi_platform_vmm_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
    set_target_properties(multi_platform_vmm_test PROPERTIES LINKER_LANGUAGE HIP)
  endif()
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()
