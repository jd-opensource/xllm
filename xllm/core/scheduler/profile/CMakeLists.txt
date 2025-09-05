
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    profile
  HDRS
    profile_manager.h
    time_predictor.h
  SRCS
    profile_manager.cpp
    time_predictor.cpp
  DEPS
    :batch
    :request
    :runtime
    glog::glog
    absl::time
)
