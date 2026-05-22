include(cc_test)

cc_test(
  NAME
    embedding_cache_test
  SRCS
    embedding_cache_test.cpp
  DEPS
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(embedding_cache_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    kv_cache_estimation_test
  SRCS
    kv_cache_estimation_test.cpp
  DEPS
    :kv_cache_estimation
    GTest::gtest_main
)

cc_test(
  NAME
    kv_cache_test
  SRCS
    kv_cache_test.cpp
  DEPS
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(kv_cache_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

# Helper test for KVCacheTransfer::BuildPushSchedule. The helper itself has
# no NPU/folly dependencies, but it lives in kv_cache_transfer.cpp alongside
# KVCacheTransferFactory::create / *_async (which DO need the full backend
# implementations + folly futures + brpc). Linking those into a standalone
# gtest binary is fragile across build envs, so this target is OFF by
# default and equivalent coverage is provided by the pure-stdlib Python
# mirror at `tools/tests/test_kv_push_schedule_cpu.py`.
# Opt in with: cmake -DXLLM_BUILD_KV_PUSH_SCHEDULE_TEST=ON ...
option(XLLM_BUILD_KV_PUSH_SCHEDULE_TEST
       "Build kv_push_schedule_test (requires :kv_cache link deps)" OFF)
if(XLLM_BUILD_KV_PUSH_SCHEDULE_TEST)
  cc_test(
    NAME
      kv_push_schedule_test
    SRCS
      kv_push_schedule_test.cpp
    DEPS
      :kv_cache
      :common
      torch
      $<$<BOOL:${USE_NPU}>:torch_npu>
      glog::glog
      gflags
      proto::xllm_proto
      GTest::gtest_main
  )
  target_link_libraries(kv_push_schedule_test
                        PUBLIC
                        Python::Python
                        $<$<BOOL:${USE_NPU}>:ascendcl>
                        $<$<BOOL:${USE_NPU}>:hccl>
                        $<$<BOOL:${USE_NPU}>:c_sec>
                        $<$<BOOL:${USE_NPU}>:nnopbase>)
endif()
