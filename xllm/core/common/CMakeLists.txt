include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    device_memory.h
    device_monitor_async_wrapper.h
    etcd_client.h
    global_flags.h
    instance_name.h
    layer_synchronizer.h
    macros.h
    metrics.h
    mspti_helper.h
    options.h
    rate_limiter.h
    types.h
    device_monitor.h
  SRCS
    device_memory.cpp
    device_monitor_async_wrapper.cpp
    etcd_client.cpp
    global_flags.cpp
    layer_synchronizer.cpp
    metrics.cpp
    mspti_helper.cpp
    options.cpp
    rate_limiter.cpp
    device_monitor.cpp
  DEPS
    util
    absl::random_random
    absl::strings
    torch
    torch_npu
    mspti
    ms_tools_ext
    Boost::serialization
    cpprest
    etcd-cpp-api
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


