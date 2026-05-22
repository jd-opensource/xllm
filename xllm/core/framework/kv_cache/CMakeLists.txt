include(cc_library)

cc_library(
  NAME
    deepseek_v4_cache_policy
  HDRS
    deepseek_v4_cache_policy.h
  DEPS
    torch
)

cc_library(
  NAME
    kv_cache_estimation
  HDRS
    kv_cache_estimation.h
  SRCS
    kv_cache_estimation.cpp
  DEPS
    :deepseek_v4_cache_policy
    :block_utils
    glog::glog
    torch
)

cc_library(
  NAME
    kv_cache
  HDRS
    deepseek_v4_kv_cache_impl.h
    embedding_cache.h
    indexed_kv_cache_impl.h
    kv_cache.h
    kv_cache_event.h
    kv_cache_impl.h
    kv_cache_shape.h
    kv_cache_tensor_role.h
    kv_cache_utils.h
    linear_attention_kv_cache_impl.h
    quantized_kv_cache_impl.h
  SRCS
    deepseek_v4_kv_cache_impl.cpp
    embedding_cache.cpp 
    indexed_kv_cache_impl.cpp
    kv_cache.cpp
    kv_cache_impl.cpp
    kv_cache_shape.cpp
    kv_cache_utils.cpp
    linear_attention_kv_cache_impl.cpp
    quantized_kv_cache_impl.cpp
  DEPS
    :block_utils
    :common
    :deepseek_v4_cache_policy
    :kv_cache_estimation
    :xtensor
    glog::glog
    torch
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:torch_npu>
)

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
