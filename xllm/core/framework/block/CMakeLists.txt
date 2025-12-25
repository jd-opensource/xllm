include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    block
  HDRS
    block.h
    block_manager.h
    block_manager_pool.h
    block_manager_impl.h
    concurrent_block_manager_impl.h
    hierarchy_block_manager_pool.h
  SRCS 
    block.cpp
    block_manager_pool.cpp
    concurrent_block_manager_impl.cpp
    block_manager_impl.cpp
    hierarchy_block_manager_pool.cpp
  DEPS
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:graph>
    :request
    :common
    glog::glog
    Boost::serialization
    SMHasherSupport
    torch
)
target_link_libraries(block PRIVATE Folly::folly)

if(USE_NPU)
  set(TEST_SRCS
    block_manager_test.cpp
  )

  cc_test(
    NAME
      block_test
    SRCS
      ${TEST_SRCS}
    DEPS
      :block
      :flags
      :kv_cache
      :prefix_cache
      absl::random_random
      Boost::serialization
      GTest::gtest_main
  )

  target_link_libraries(block_test PRIVATE OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
  add_dependencies(block_test brpc-static)
endif()
