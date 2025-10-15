include(cc_library)

cc_library(
  NAME
    xllm_server
  HDRS
    xllm_server.h
    xllm_server_registry.h
  SRCS
    xllm_server.cpp
    xllm_server_registry.cpp
  DEPS
    :api_service
    :request
    absl::strings
    glog::glog
    proto::xllm_proto
)
