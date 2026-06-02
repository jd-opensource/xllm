include(cc_test)

cc_test(
  NAME
    activation_dcu_test
  SRCS
    activation_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(activation_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    norm_dcu_test
  SRCS
    norm_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(norm_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    rope_dcu_test
  SRCS
    rope_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(rope_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

option(XLLM_ENABLE_DCU_BLOCK_COPY_TEST
       "Build and register the expensive DCU block_copy_test"
       OFF
)

if(XLLM_ENABLE_DCU_BLOCK_COPY_TEST)
  cc_test(
    NAME
      block_copy_dcu_test
    SRCS
      block_copy_test.cpp
    DEPS
      :dcu_kernels
      hip::host
      torch
      GTest::gtest_main
      glog::glog
  )
  target_compile_definitions(block_copy_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
endif()

cc_test(
  NAME
    fused_qknorm_rope_dcu_test
  SRCS
    fused_qknorm_rope_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(fused_qknorm_rope_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    matmul_dcu_test
  SRCS
    matmul_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(matmul_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    air_log_softmax_dcu_test
  SRCS
    air_log_softmax_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(air_log_softmax_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    reshape_paged_cache_dcu_test
  SRCS
    reshape_paged_cache_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(reshape_paged_cache_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})

cc_test(
  NAME
    random_sample_dcu_test
  SRCS
    random_sample_test.cpp
  DEPS
    :dcu_kernels
    hip::host
    torch
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(random_sample_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
