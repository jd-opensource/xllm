include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    etcd_client.h
    global_flags.h
    instance_name.h
    macros.h
    metrics.h
    $<$<BOOL:${USE_NPU}>:mspti_helper.h>
    options.h
    rate_limiter.h
    types.h
    device_monitor.h
    version_singleton.h
  SRCS
    etcd_client.cpp
    global_flags.cpp
    metrics.cpp
    $<$<BOOL:${USE_NPU}>:mspti_helper.cpp>
    options.cpp
    rate_limiter.cpp
    device_monitor.cpp
  DEPS
    util
    absl::random_random
    absl::strings
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_MSPTI}>:mspti>
    $<$<BOOL:${USE_NPU}>:ms_tools_ext>
    Boost::serialization
    cpprest
    etcd-cpp-api
    $<$<BOOL:${USE_MLU}>:torch_mlu>
)

cc_library(
  NAME
    flags
  HDRS
    global_flags.h
  SRCS
    global_flags.cpp
  DEPS
    gflags::gflags
)

cc_test(
  NAME 
    common_test
  SRCS
    rate_limiter_test.cpp
  DEPS
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
)
target_link_libraries(common PRIVATE OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(common brpc-static)


