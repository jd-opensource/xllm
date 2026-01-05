include(cc_library)

cc_library(
  NAME
    cuda_layers
  HDRS
    attention.h
    flashinfer_workspace.h
    flashinfer_planinfo.h
  SRCS
    attention.cpp
    flashinfer_workspace.cpp
    flashinfer_planinfo.cpp
  DEPS
    :common_layers
)
