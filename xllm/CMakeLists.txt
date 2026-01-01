include(cc_binary)
include(cc_shared_library)

include_directories(.)

# Set warning-as-error for xllm code (third-party headers are marked as SYSTEM)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:C>:-Werror>
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
      $<$<COMPILE_LANGUAGE:C>:-Wno-macro-redefined>
    )
  endif()
endif()

option(GENERATE_SO "Enable compile to generate .so" OFF)

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(parser)
add_subdirectory(processors)
add_subdirectory(proto)
add_subdirectory(pybind)
add_subdirectory(server)

if(GENERATE_SO)
  message(STATUS "build libxllm.so and install")
  cc_shared_library(
    NAME
      xllm
    HDRS
      cc_api/llm.h
      cc_api/types.h
      cc_api/internal.h
    SRCS
      cc_api/llm.cpp
    DEPS
      :flags
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

else()
  message(STATUS "build xllm binary and install")
  cc_binary(
    NAME
      xllm
    SRCS
      xllm.cpp
    DEPS
      :flags
      :xllm_server
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

endif()

# link brpc
target_link_libraries(xllm PRIVATE glog::glog brpc leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf ${OpenCV_LIBS})
add_dependencies(xllm brpc-static)

if(USE_NPU)
  set(COMMON_LIBS Python::Python ascendcl atb_customize hccl c_sec nnopbase ms_tools_ext torch_npu torch_python)
elseif(USE_MLU)
  set(COMMON_LIBS Python::Python)
endif()

if(USE_MSPTI)
  list(APPEND COMMON_LIBS mspti)
endif()
target_link_libraries(xllm PUBLIC ${COMMON_LIBS})

# install xllm
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install all dependencies for xllm
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
        RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
        UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
        EXECUTABLES $<TARGET_FILE:xllm>)

  file(INSTALL
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        FILES ${DEPENDENCIES}
        FOLLOW_SYMLINK_CHAIN)

  # This should not be possible, but error out when a dependency cannot
  # be resolved.
  list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
  if(${UNRESOLVED_LENGTH} GREATER 0)
      message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
  endif()
]])
