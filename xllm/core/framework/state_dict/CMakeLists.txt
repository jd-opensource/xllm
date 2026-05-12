include(cc_test)
include(cc_library)

include_directories(..)

add_subdirectory(safetensors)

cc_library(
  NAME 
    state_dict
  HDRS
    state_dict.h
    utils.h
    rec_vocab_dict.h
  SRCS
    state_dict.cpp
    utils.cpp
    rec_vocab_dict.cpp
  DEPS
    rust_safetensors
    torch
    glog::glog
    Folly::folly
    util
)

cc_test(
  NAME
    rec_vocab_dict_test
  SRCS
    rec_vocab_dict_test.cpp
  DEPS
    :flags
    :state_dict
    GTest::gtest_main
    glog::glog
)
