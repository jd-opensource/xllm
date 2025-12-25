include(cc_library)

add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    npu_ops_api.h
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    matmul.cpp
    rope.cpp
  DEPS
    :torch_npu_kernels
)
