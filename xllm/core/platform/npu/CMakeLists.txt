include(cc_library)

cc_library(
  NAME 
    platform_npu
  HDRS
    layer_synchronizer.h
    npu_layer_synchronizer.h
    device_capture_lock.h
  SRCS
    layer_synchronizer.cpp
    npu_layer_synchronizer.cpp
  DEPS
    torch_npu
    glog::glog
    torch
    ascendcl
)