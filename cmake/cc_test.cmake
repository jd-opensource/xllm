include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_test()
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# LINKOPTS: List of link options
# ARGS: Command line arguments to test case
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#
# cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     :awesome
#     GTest::gmock
# )
#
function(cc_test)
  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    CC_TEST # prefix
    "" # options
    "NAME" # one value args
    "SRCS;COPTS;LINKOPTS;DEPS;INCLUDES;ARGS;DATA" # multi value args
    ${ARGN}
  )

  # place test data in build directory
  if(CC_TEST_DATA)
    foreach(data ${CC_TEST_DATA})
      configure_file(${data} ${CMAKE_CURRENT_BINARY_DIR}/${data} COPYONLY)
    endforeach()
  endif()

  add_executable(${CC_TEST_NAME})
  target_sources(${CC_TEST_NAME} PRIVATE ${CC_TEST_SRCS})
  target_include_directories(${CC_TEST_NAME}
    PUBLIC 
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>" 
      ${CC_TEST_INCLUDES}
  )

  target_compile_options(${CC_TEST_NAME}
    PRIVATE ${CC_TEST_COPTS}
  )

  target_link_libraries(${CC_TEST_NAME}
    PUBLIC ${CC_TEST_DEPS}
    PRIVATE ${CC_TEST_LINKOPTS}
  )

  if(USE_NPU)
    set(COMMON_LIBS Python::Python torch_npu torch_python)
    target_link_libraries(${CC_TEST_NAME} PRIVATE ${COMMON_LIBS})
  endif()

  add_dependencies(all_tests ${CC_TEST_NAME})

  gtest_add_tests(
    TARGET ${CC_TEST_NAME}
    EXTRA_ARGS ${CC_TEST_ARGS}
  )
  #add_test(NAME ${CC_TEST_NAME} COMMAND ${CC_TEST_NAME} ${CC_TEST_ARGS})
endfunction()
