include(cc_test)

# Add test for common test
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_tests.cpp
    tests_utils.cpp
    deepseek_v32_sp_utils_tests.cpp
    dp_utils_tests.cpp
    indexer_tests.cpp
    mla_tests.cpp
    deepseek_v2_decoder_layer_tests.cpp
    qwen2_attention_tests.cpp
    qwen2_vision_attention_tests.cpp
  DEPS
    :common_layers
    :mlu_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

# MoE Gate unit test for MLU (uses layers/mlu/moe_gate)
cc_test(
  NAME
    moe_layer_test
  SRCS
    moe_gate_tests.cpp
    fused_moe_tests.cpp
    deepseek_v2_sparse_moe_block_tests.cpp
    tests_utils.cpp
  DEPS
    :common_layers
    :mlu_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

# Add test for DeepEP
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    deep_ep_test
  SRCS
    deep_ep_tests.cpp
    tests_utils.cpp
  DEPS
    :common_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

# Add test for FusedMoE All2All path
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    fused_moe_all2all_test
  SRCS
    fused_moe_all2all_tests.cpp
    tests_utils.cpp
  DEPS
    :common_layers
    :mlu_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

cc_test(
  NAME
    deepseek_v2_attention_multi_device_test
  SRCS
    deepseek_v2_attention_multi_device_tests.cpp
    tests_utils.cpp
  DEPS
    :block
    :mlu_layers
    :parallel_state
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)
