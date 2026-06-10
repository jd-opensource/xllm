include(cc_library)

cc_library(
  NAME 
    platform_npu
  HDRS
    npu_layer_synchronizer.h
    device_capture_lock.h
    npu_batch_memcpy.h
  SRCS
    npu_layer_synchronizer.cpp
    npu_batch_memcpy.cpp
  DEPS
    torch_npu
    glog::glog
    torch
    ascendcl
)
