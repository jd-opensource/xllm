include(cc_test)

cc_test(
  NAME
    moe_fused_topk_test
  SRCS
    moe_topk_test.cu
  DEPS
    :cuda_kernels
    GTest::gtest_main
)
