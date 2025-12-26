include(cc_library)
include(cc_test)

add_subdirectory(partial_json_parser)

cc_library (
  NAME
    function_call
  HDRS
        core_types.h
        base_format_detector.h
        qwen25_detector.h
        kimik2_detector.h
        deepseekv3_detector.h
        glm45_detector.h
        glm47_detector.h
        function_call_parser.h
        function_call.h
        utils.h
  SRCS
        base_format_detector.cpp
        qwen25_detector.cpp
        kimik2_detector.cpp
        deepseekv3_detector.cpp
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

function(add_detector_test TEST_NAME)
  cc_test(
    NAME
      ${TEST_NAME}
    SRCS
      ${TEST_NAME}.cpp
    DEPS
      :function_call
      GTest::gtest
      GTest::gtest_main
      nlohmann_json::nlohmann_json
  )
endfunction()

add_detector_test(qwen25_detector_test)
add_detector_test(kimik2_detector_test)
add_detector_test(deepseekv3_detector_test)
add_detector_test(glm45_detector_test)
add_detector_test(glm47_detector_test)

