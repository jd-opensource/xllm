include(cc_library)
include(cc_test)

add_subdirectory(tokenizers)

cc_library(
  NAME 
    tokenizer
  HDRS
    tokenizer_args.h
    tokenizer.h
    tokenizer_factory.h
    tiktoken_tokenizer.h
    sentencepiece_tokenizer.h
    fast_tokenizer.h
    tokenizer_proxy.h
    rec_tokenizer.h
  SRCS
    tokenizer_factory.cpp
    tiktoken_tokenizer.cpp
    sentencepiece_tokenizer.cpp
    fast_tokenizer.cpp
    tokenizer_proxy.cpp
    rec_tokenizer.cpp
  DEPS
    :common
    :sentencepiece
    absl::flat_hash_map
    absl::strings
    glog::glog
    rust_tokenizers
    re2::re2
)

cc_test(
  NAME
    fast_tokenizer_test
  SRCS
    tests/fast_tokenizer_tests.cpp
  DEPS
    :tokenizer
    glog::glog
    GTest::gtest_main
)

