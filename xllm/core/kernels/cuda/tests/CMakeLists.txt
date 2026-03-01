include(cc_test)

# Common link options for brpc static library dependency resolution.
# :util (dep of :cuda_kernels) depends on brpc, which depends on OpenSSL and leveldb.
# When using static libraries, link order matters. Use LINK_GROUP with RESCAN
# to let the linker rescan the group until all symbols are resolved.
set(BRPC_LINK_GROUP "$<LINK_GROUP:RESCAN,brpc,leveldb::leveldb,OpenSSL::SSL,OpenSSL::Crypto>")

# Common dependencies for cuda kernel tests
set(CUDA_KERNEL_TEST_DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)

# cutlass_scaled_mm_test
cc_test(
  NAME
    cutlass_scaled_mm_test
  SRCS
    cutlass_scaled_mm_test.cpp
  DEPS
    ${CUDA_KERNEL_TEST_DEPS}
  LINKOPTS
    ${BRPC_LINK_GROUP}
)
add_dependencies(cutlass_scaled_mm_test brpc-static)

# static_scaled_fp8_quant_test
cc_test(
  NAME
    static_scaled_fp8_quant_test
  SRCS
    static_scaled_fp8_quant_test.cpp
  DEPS
    ${CUDA_KERNEL_TEST_DEPS}
  LINKOPTS
    ${BRPC_LINK_GROUP}
)
add_dependencies(static_scaled_fp8_quant_test brpc-static)
