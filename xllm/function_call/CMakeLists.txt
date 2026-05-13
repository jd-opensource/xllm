include(cc_library)

add_subdirectory(partial_json_parser)

cc_library (
  NAME
    function_call
  HDRS
        core_types.h
        base_format_detector.h
        qwen25_detector.h
        qwen3_coder_detector.h
        kimik2_detector.h
        deepseekv3_detector.h
        deepseekv32_detector.h
        glm45_detector.h
        glm47_detector.h
        function_call_parser.h
        function_call.h
        utils.h
  SRCS
        base_format_detector.cpp
        qwen25_detector.cpp
        qwen3_coder_detector.cpp
        kimik2_detector.cpp
        deepseekv3_detector.cpp
        deepseekv32_detector.cpp
        glm45_detector.cpp
        glm47_detector.cpp
        function_call_parser.cpp
        utils.cpp
  DEPS
    nlohmann_json::nlohmann_json
    glog::glog
    proto::xllm_proto
    partial_json_parser
    common
)
