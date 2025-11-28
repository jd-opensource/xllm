include(cc_library)
include(cc_test)

include_directories(
  ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
)


cc_library(
  NAME
    npu_kernels_impl
  HDRS
    npu_split_impl.h
    npu_linear_impl.h
    npu_rms_norm_impl.h
    npu_rope_impl.h
  SRCS
    npu_split_impl.cpp
    npu_linear_impl.cpp
    npu_rms_norm_impl.cpp
    npu_rope_impl.cpp
  DEPS
    :npu_layers 
    :model_context
    :state_dict
    glog::glog
    torch
    torch_npu
)
