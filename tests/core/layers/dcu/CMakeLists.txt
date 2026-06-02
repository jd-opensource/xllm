include(cc_test)

cc_test(
  NAME
    activation_layer_dcu_test
  SRCS
    activation_test.cpp
  DEPS
    :common_layers
    :platform
    glog::glog
    torch
    GTest::gtest_main
)
target_compile_definitions(activation_layer_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
