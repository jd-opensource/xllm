include(cc_library)

add_subdirectory(impl)
add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    linear.h
    split.h
    rms_norm.h
    rope.h
  DEPS
    :npu_kernels_impl
    # spdlog::spdlog
)