include(cc_library)

cc_library(
  NAME
    ilu_layers
  HDRS
    attention.h
    flashinfer_workspace.h
  SRCS
    attention.cpp
    flashinfer_workspace.cpp
  DEPS
    :common_layers
)
