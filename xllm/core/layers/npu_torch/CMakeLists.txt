include(cc_library)

cc_library(
  NAME
    npu_torch_layers
  HDRS
    attention.h
  SRCS
    attention.cpp
  DEPS
    :common_layers
)
