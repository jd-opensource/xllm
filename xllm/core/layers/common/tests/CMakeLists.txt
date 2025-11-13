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
    deepseek_v2_decoder_layer_tests.cpp
    tests_utils.cpp
  DEPS
    :common_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)
