include(cc_test)

# Add test for common test
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_tests.cpp
    tests_utils.cpp
    indexer_tests.cpp
    mla_tests.cpp
    deepseek_v2_decoder_layer_tests.cpp
    qwen2_attention_tests.cpp
    qwen2_vision_attention_tests.cpp
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

# MoE Gate unit test for MLU (uses layers/mlu/moe_gate)
if(USE_MLU)
cc_test(
  NAME
    moe_layer_test
  SRCS
    moe_gate_tests.cpp
    fused_moe_tests.cpp
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
endif()

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
