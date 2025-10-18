include(cc_library)
include(cc_test)


add_subdirectory(profile)

cc_library(
  NAME 
    scheduler
  HDRS
    chunked_prefill_scheduler.h
    zero_eviction_scheduler.h
    continuous_scheduler.h
    disagg_pd_scheduler.h
    async_response_processor.h
    scheduler.h
    dit_scheduler.h
    scheduler_factory.h
    decode_priority_queue.h
    pure_chunked_scheduler.h
  SRCS
    chunked_prefill_scheduler.cpp
    zero_eviction_scheduler.cpp
    continuous_scheduler.cpp
    disagg_pd_scheduler.cpp
    async_response_processor.cpp
    dit_scheduler.cpp
    scheduler_factory.cpp
    pure_chunked_scheduler.cpp
  DEPS
    :batch
    :request
    :runtime
    :profile
    glog::glog
    Folly::folly
    absl::time
    absl::synchronization
)

cc_test(
  NAME
    chunked_prefill_scheduler_test
    continuous_scheduler_test
  SRCS
    chunked_prefill_scheduler_test.cpp
    continuous_scheduler_test.cpp
  DEPS
    :scheduler
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)

