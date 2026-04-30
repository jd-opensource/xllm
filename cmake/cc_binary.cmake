include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_binary()
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# DEPS: List of other libraries to be linked in to the binary targets
#
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# cc_binary(
#   NAME
#     fantastic
#   SRCS
#     "b.cc"
#   DEPS
#     :awesome
# )
#
function(cc_binary)
  cmake_parse_arguments(
    CC_BINARY # prefix
    "" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS" # multi value args
    ${ARGN}
  )

  add_executable(${CC_BINARY_NAME} "")
  target_sources(${CC_BINARY_NAME} 
    PRIVATE ${CC_BINARY_SRCS} ${CC_BINARY_HDRS}
  )
  target_link_libraries(${CC_BINARY_NAME}
    PUBLIC 
      ${CC_BINARY_DEPS}
    PRIVATE
      ${CC_BINARY_LINKOPTS}
  )
  target_include_directories(${CC_BINARY_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"      
  )
  target_compile_options(${CC_BINARY_NAME} PRIVATE ${CC_BINARY_COPTS})
  target_compile_definitions(${CC_BINARY_NAME} PUBLIC ${CC_BINARY_DEFINES})
  set_target_properties(${CC_BINARY_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )

  add_executable(:${CC_BINARY_NAME} ALIAS ${CC_BINARY_NAME})
endfunction()
