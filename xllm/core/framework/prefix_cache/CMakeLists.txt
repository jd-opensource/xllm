include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    prefix_cache
  HDRS
    prefix_cache_hash.h
    prefix_cache_hash_with_upload.h
  SRCS 
    prefix_cache.cpp
    prefix_cache_hash.cpp
    prefix_cache_hash_with_upload.cpp
  DEPS
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:llm_engine>
    $<$<BOOL:${USE_NPU}>:graph>
    :request
    :common
    glog::glog
    Boost::serialization
    SMHasherSupport
    torch
)

set(TEST_SRCS
  prefix_cache_hash_test.cpp
)

cc_test(
  NAME
    prefix_test
  SRCS
    ${TEST_SRCS}
  DEPS
    :flags
    :kv_cache
    :prefix_cache
    :block
    absl::random_random
    Boost::serialization
    GTest::gtest_main
)

target_link_libraries(prefix_test PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto ascendcl Folly::folly)
add_dependencies(prefix_test brpc-static)

cc_binary(
  NAME
    prefix_cache_benchmark
  SRCS
    prefix_cache_benchmark.cpp
  DEPS
    :kv_cache
    :prefix_cache
    :block
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(prefix_cache_benchmark PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(prefix_cache_benchmark brpc-static)
