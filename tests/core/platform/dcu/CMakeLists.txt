include(cc_test)

cc_test(
  NAME
    vmm_api_dcu_test
  SRCS
    vmm_api_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(vmm_api_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(vmm_api_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    stream_dcu_test
  SRCS
    stream_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(stream_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(stream_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    device_dcu_test
  SRCS
    device_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(device_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(device_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    shared_vmm_allocator_dcu_test
  SRCS
    shared_vmm_allocator_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(shared_vmm_allocator_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(shared_vmm_allocator_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    numa_utils_dcu_test
  SRCS
    numa_utils_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(numa_utils_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(numa_utils_dcu_test PROPERTIES LINKER_LANGUAGE HIP)
