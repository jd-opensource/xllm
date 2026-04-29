include(cc_library)


add_subdirectory(profile)

cc_library(
  NAME
    scheduler
  HDRS
    mix_scheduler.h
    chunked_prefill_scheduler.h
    zero_eviction_scheduler.h
    continuous_scheduler.h
    disagg_pd_scheduler.h
    pd_ooc_scheduler.h
    async_response_processor.h
    scheduler.h
    dit_scheduler.h
    prefill_only_scheduler.h
    scheduler_factory.h
    decode_priority_queue.h
    perf_model.h
    fixed_steps_scheduler.h
  SRCS
    mix_scheduler.cpp
    chunked_prefill_scheduler.cpp
    zero_eviction_scheduler.cpp
    continuous_scheduler.cpp
    disagg_pd_scheduler.cpp
    pd_ooc_scheduler.cpp
    async_response_processor.cpp
    dit_scheduler.cpp
    prefill_only_scheduler.cpp
    scheduler_factory.cpp
    perf_model.cpp
    fixed_steps_scheduler.cpp
  DEPS
    :batch
    :request
    :runtime
    :profile
    :pd_topology_guard
    glog::glog
    Folly::folly
    absl::time
    absl::synchronization
)



# Use LINK_GROUP to resolve circular dependency between xtensor and xllm_server
# This wraps the libraries in --start-group/--end-group
