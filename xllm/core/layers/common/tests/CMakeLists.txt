include(cc_test)

# Add test for DenseMLP
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_tests.cpp
    fused_moe_tests.cpp
    indexer_tests.cpp
    mla_tests.cpp
    tests_utils.cpp
  DEPS
    :common_layers
    :parallel_state
    :model
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)
