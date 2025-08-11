include(cc_library)
include(cc_test)

cc_library (
  NAME
    function_call
  HDRS
        core_types.h
        base_format_detector.h
        qwen25_detector.h
        function_call_parser.h
        function_call.h
  SRCS
        base_format_detector.cpp
        qwen25_detector.cpp
        function_call_parser.cpp
  DEPS
    nlohmann_json::nlohmann_json
    glog::glog
    proto::xllm_proto
)
