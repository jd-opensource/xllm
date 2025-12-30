include(cc_test)

if(USE_MLU)
	list(APPEND TEST_SRCS indexer_tests.cpp mla_tests.cpp deepseek_v2_decoder_layer_tests.cpp)
endif()

if(NOT USE_NPU)
	list(APPEND TEST_SRCS fused_moe_tests.cpp)
endif()

# Add test for common test
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_tests.cpp
    tests_utils.cpp
    ${TEST_SRCS}
  DEPS
    :common_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    $<$<BOOL:${USE_NPU}>:atb>
    $<$<BOOL:${USE_NPU}>:atb_customize>
    $<$<BOOL:${USE_NPU}>:nnopbase>
    $<$<BOOL:${USE_NPU}>:c_sec>
    glog::glog
    torch
    GTest::gtest_main
)

# Add test for DeepEP
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
if(USE_MLU)
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
    $<$<BOOL:${USE_NPU}>:atb>
    $<$<BOOL:${USE_NPU}>:atb_customize>
    $<$<BOOL:${USE_NPU}>:nnopbase>
    GTest::gtest_main
    torch
    glog::glog
)
endif()
