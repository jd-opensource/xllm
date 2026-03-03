include(cc_library)

cc_library(
  NAME
    npu_torch_layers
  HDRS
    fused_moe.h
    attention.h
  SRCS
    fused_moe.cpp
    attention.cpp
  DEPS
    :common_layers
)
