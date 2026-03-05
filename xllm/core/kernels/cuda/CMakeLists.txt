include(cc_library)
include(cc_test)

# CUTLASS headers are provided by root CMakeLists.txt via submodule (third_party/cutlass)
include_directories(${CMAKE_CURRENT_LIST_DIR})

#
# CUDA feature detection
#
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  # SM90+ (Hopper/Blackwell) scaled matmul support
  add_compile_definitions(
    ENABLE_SCALED_MM_SM90=1
    ENABLE_SCALED_MM_SM100=1
    ENABLE_SCALED_MM_SM120=1
  )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  add_compile_definitions(ENABLE_FP8)
endif()

set(CUDA_HEADER_FILES
  arch_condition.h
  attention_runner.h
  cuda_ops_api.h
  cuda_utils.h
  cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
  cutlass_w8a8/c3x/scaled_mm.cuh
  cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh
  fp8_quant_utils.cuh
  global_capture_instance.h
  moe_topk_funcs.cuh
  piecewise_graphs.h
  topk_last_dim.cuh
  type_convert.cuh
  utils.h
  xattention/xattention_ops_api.h
)

# Keep source list explicit to avoid accidentally compiling test files into cuda_kernels.
set(CUDA_SOURCE_FILES
  activation.cu
  air_log_softmax_last_dim.cu
  attention_runner.cpp
  batch_decode.cpp
  batch_prefill.cpp
  cutlass_extensions/common.cpp
  cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu
  cutlass_w8a8/c3x/scaled_mm_sm120_fp8.cu
  cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu
  cutlass_w8a8/scaled_mm_c3x_sm100.cu
  cutlass_w8a8/scaled_mm_c3x_sm120.cu
  cutlass_w8a8/scaled_mm_c3x_sm90.cu
  cutlass_w8a8/scaled_mm_entry.cu
  fp8_quant.cu
  fp8_scaled_matmul.cpp
  fp8_scaled_quantize.cpp
  fused_qknorm_rope.cu
  global_capture_instance.cpp
  matmul.cpp
  norm.cu
  piecewise_graphs.cpp
  rec_beam_search.cu
  reshape_paged_cache.cu
  rope.cu
  utils.cpp
  xattention/beam_search.cpp
  xattention/cache_select.cu
  xattention/decoder_reshape_and_cache.cu
  xattention/lse_combine.cu
  xattention/prefill_reshape_and_cache.cu
)

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${CUDA_HEADER_FILES}
  SRCS
    ${CUDA_SOURCE_FILES}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
)

cc_test(
  NAME
    decoder_reshape_and_cache_test
  SRCS
    xattention/decoder_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(decoder_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    prefill_reshape_and_cache_test
  SRCS
    xattention/prefill_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(prefill_reshape_and_cache_test PRIVATE brpc)

cc_test(
  NAME
    cache_select_test
  SRCS
    xattention/cache_select_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cache_select_test PRIVATE brpc)

cc_test(
  NAME
    moe_topk_funcs_test
  SRCS
    moe_topk_funcs_test.cu
  DEPS
    :cuda_kernels
    GTest::gtest_main
)

cc_test(
  NAME
    fused_qknorm_rope_test
  SRCS
    fused_qknorm_rope_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    cutlass_scaled_mm_test
  SRCS
    cutlass_scaled_mm_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cutlass_scaled_mm_test PRIVATE brpc)

cc_test(
  NAME
    fp8_quant_test
  SRCS
    fp8_quant_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(fp8_quant_test PRIVATE brpc)
