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
  cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
  cutlass_w8a8/c3x/scaled_mm.cuh
  cutlass_w8a8/c3x/sm120_fp8_kernel_configs.cuh
  cutlass_w8a8/c3x/sm120_fp8_dispatch_policy.hpp
  cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh
  fp8_quant_utils.cuh
  global_capture_instance.h
  piecewise_graphs.h
  topk_last_dim.cuh
  type_convert.cuh
  utils.h
  device_utils.cuh
  xattention/xattention_ops_api.h
  moe/moe_topk.cuh
  moe/moe_topk_sigmoid_kernels.cuh
  moe/moe_topk_softmax_kernels.cuh
)

#
# Per-SM-architecture CUTLASS libraries.
#
# CUTLASS template instantiations are extremely expensive to compile. Splitting each SM generation
# into its own static library improves incremental build times — changing one SM's CUTLASS code
# only recompiles that library, not all CUTLASS files.

set(_CUTLASS_SM_LIBS "")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  # --- SM90 (Hopper) ---
  cc_library(
    NAME cutlass_sm90
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm90.cu
    DEPS
      torch
      tvm_ffi
  )

  # --- SM100 (Blackwell) ---
  cc_library(
    NAME cutlass_sm100
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm100.cu
    DEPS
      torch
      tvm_ffi
  )

  # --- SM120 ---
  cc_library(
    NAME cutlass_sm120
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm120_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm120.cu
    DEPS
      torch
      tvm_ffi
  )

  set(_CUTLASS_SM_LIBS :cutlass_sm90 :cutlass_sm100 :cutlass_sm120)
endif()

# Keep source list explicit to avoid accidentally compiling test files into cuda_kernels.
# NOTE: SM-specific CUTLASS sources are compiled via the per-arch libraries above.
set(CUDA_SOURCE_FILES
  activation.cu
  air_log_softmax_last_dim.cu
  attention_runner.cpp
  batch_decode.cpp
  batch_prefill.cpp
  batch_chunked_prefill.cpp
  block_copy.cu
  cutlass_extensions/common.cpp
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
  random_sample.cpp
  utils.cpp
  xattention/beam_search.cpp
  xattention/cache_select.cu
  xattention/decoder_reshape_and_cache.cu
  xattention/lse_combine.cu
  xattention/prefill_reshape_and_cache.cu
  moe/moe_fused_topk.cu
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
    ${_CUTLASS_SM_LIBS}
)

cc_test(
  NAME
    activation_test
  SRCS
    activation_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)

option(XLLM_ENABLE_BLOCK_COPY_TEST
       "Build and register the expensive CUDA block_copy_test"
       OFF)

if(XLLM_ENABLE_BLOCK_COPY_TEST)
  cc_test(
    NAME
      block_copy_test
    SRCS
      block_copy_test.cpp
    DEPS
      :cuda_kernels
      torch
      GTest::gtest_main
      glog::glog
  )
endif()

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
    moe_fused_topk_test
  SRCS
    moe/moe_topk_test.cu
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

# Keep CUDA benchmarks behind a plain CMake cache option instead of adding
# another dedicated argument in `setup.py`.
#
# Why:
# - `setup.py` already forwards arbitrary CMake definitions from the
#   `CMAKE_ARGS` environment variable, so adding a new Python-side flag would
#   duplicate the existing mechanism.
# - Most normal builds do not need benchmark binaries, so the default stays
#   `OFF` and users enable it only when they really want to build them.
#
# Simplest way to build benchmarks:
# 1. From the repo root, run:
#      CMAKE_ARGS="-DBUILD_CUDA_BENCHMARKS=ON" python setup.py build test
# 2. This is enough: `build test` builds `all_tests`, and when
#    `BUILD_CUDA_BENCHMARKS=ON` the benchmark binaries are added to
#    `all_tests` as build dependencies.
# 3. If the build directory was configured before with this option `OFF`, the
#    old value is cached. Re-run the same command, or delete `CMakeCache.txt`
#    / the build directory before rebuilding.
option(BUILD_CUDA_BENCHMARKS "Build CUDA kernel benchmarks" OFF)
if(BUILD_CUDA_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
