include(cc_library)

add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    linear.h
    split.h
    rope.h
  DEPS
    # spdlog::spdlog
)
