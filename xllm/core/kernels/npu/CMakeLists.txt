include(cc_library)

add_subdirectory(impl)
add_subdirectory(xllm_ops)
add_subdirectory(triton)

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
    # triton_adapter
    # spdlog::spdlog
)
