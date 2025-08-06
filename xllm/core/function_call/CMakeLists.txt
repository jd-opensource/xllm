cc_library(
  NAME function_call
  HDRS
    types.h
    base_detector.h
    qwen25_detector.h
    function_call_parser.h
    function_call.h
  SRCS
    base_detector.cpp
    function_call_parser.cpp
  DEPS
    nlohmann_json::nlohmann_json
    glog::glog
)