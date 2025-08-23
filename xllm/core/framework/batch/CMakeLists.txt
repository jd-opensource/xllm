include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    batch
  HDRS
    batch.h
    batch_factory.h
    batch_input_builder.h
    mposition.h
  SRCS 
    batch.cpp
    batch_factory.cpp
    batch_input_builder.cpp
    mposition.cpp
  DEPS
    :request
    :runtime
    :common
    glog::glog
)

cc_test(
  NAME
    batch_test
  SRCS
    batch_test.cpp
  DEPS
    :batch
    absl::time
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:torch_npu>
)
target_link_libraries(batch_test PUBLIC Python::Python ascendcl hccl c_sec nnopbase)

