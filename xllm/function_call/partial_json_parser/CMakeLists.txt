include(cc_library)
include(cc_test)

cc_library(
  NAME 
    partial_json_parser
  SRCS 
    src/parser.cpp
  INCLUDES 
    include
  DEPS
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    partial_json_parser_test
  SRCS
    test/test_examples.cpp
    test/test_property_based.cpp
  DEPS
    :partial_json_parser
    GTest::gtest
    GTest::gtest_main
)