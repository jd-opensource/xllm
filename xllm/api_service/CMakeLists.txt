include(cc_library)
include(cc_test)

cc_library(
  NAME
    api_service
  HDRS
    api_service.h
    api_service_impl.h
    call.h
    chat_json_utils.h
    completion_service_impl.h
    rec_completion_service_impl.h
    chat_service_impl.h
    anthropic_service_impl.h
    embedding_service_impl.h
    image_generation_service_impl.h
    rerank_service_impl.h
    qwen3_rerank_service_impl.h
    non_stream_call.h
    service_impl_factory.h
    stream_call.h
    models_service_impl.h
    stream_output_parser.h
    mm_service_utils.h
    embedding_output_builder.h
    utils.h
  SRCS
    api_service.cpp
    call.cpp
    completion_service_impl.cpp
    rec_completion_service_impl.cpp
    chat_service_impl.cpp
    anthropic_service_impl.cpp
    embedding_service_impl.cpp
    image_generation_service_impl.cpp
    models_service_impl.cpp
    rerank_service_impl.cpp
    stream_output_parser.cpp
    qwen3_rerank_service_impl.cpp
    embedding_output_builder.cpp
  DEPS
    :master
    :chat_template
    :util
    glog::glog
    proto::xllm_proto
    absl::flat_hash_set
    absl::random_random
    :function_call
    :reasoning
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
)

cc_test(
  NAME
    anthropic_protocol_test
  SRCS
    anthropic_protocol_test.cpp
  DEPS
    proto::xllm_proto
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    glog::glog
)

# Integration test for Anthropic service - requires running server
# Tests are prefixed with DISABLED_ in code to skip by default
# Run manually: ./anthropic_service_test --gtest_also_run_disabled_tests
cc_test(
  NAME
    anthropic_service_test
  SRCS
    anthropic_service_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    glog::glog
)
target_link_libraries(anthropic_service_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(anthropic_service_test brpc-static)

cc_test(
  NAME
    api_service_test
  SRCS
    chat_json_utils_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
