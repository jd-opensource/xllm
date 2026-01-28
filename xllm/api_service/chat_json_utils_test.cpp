/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "chat_json_utils.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

namespace xllm {

class PreprocessChatJsonTest : public ::testing::Test {
 protected:
  // Helper to check successful preprocessing
  void ExpectSuccess(const std::string& input,
                     bool is_multimodal,
                     const std::string& expected_output) {
    auto [status, result] = preprocess_chat_json(input, is_multimodal);
    ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();
    // Parse both to compare JSON structure, not string equality
    auto result_json = nlohmann::json::parse(result);
    auto expected_json = nlohmann::json::parse(expected_output);
    EXPECT_EQ(result_json, expected_json);
  }

  // Helper to check expected error
  void ExpectError(const std::string& input,
                   bool is_multimodal,
                   const std::string& expected_error_substring) {
    auto [status, result] = preprocess_chat_json(input, is_multimodal);
    ASSERT_FALSE(status.ok()) << "Expected error but got success";
    EXPECT_NE(status.message().find(expected_error_substring),
              std::string::npos)
        << "Error message '" << status.message()
        << "' does not contain expected substring '" << expected_error_substring
        << "'";
  }
};

// =============================================================================
// Basic functionality tests
// =============================================================================

TEST_F(PreprocessChatJsonTest, PassThroughNonArrayContent) {
  // String content should pass through unchanged
  std::string input = R"({
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  ExpectSuccess(input, /*is_multimodal=*/false, input);
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, PassThroughNoMessages) {
  // JSON without messages field should pass through
  std::string input = R"({"model": "test"})";
  ExpectSuccess(input, /*is_multimodal=*/false, input);
}

TEST_F(PreprocessChatJsonTest, CombineTextArrayIntoString) {
  // Array of text items should be combined into single string for
  // non-multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"}
      ]
    }]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "Hello\nWorld"}]
  })";
  ExpectSuccess(input, /*is_multimodal=*/false, expected);
  // For multimodal, array is preserved (not combined)
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, SingleTextItemCombined) {
  // Single text item in array should be converted to string for non-multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  ExpectSuccess(input, /*is_multimodal=*/false, expected);
  // For multimodal, array is preserved
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

// =============================================================================
// Multimodal content tests (Issue #801)
// =============================================================================

TEST_F(PreprocessChatJsonTest, ImageUrlPassesThroughOnMultimodal) {
  // image_url content should pass through unchanged on multimodal endpoint
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "text", "text": "What is this?"}
      ]
    }]
  })";
  // Should pass through unchanged for multimodal
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, ImageUrlErrorsOnTextOnly) {
  // image_url content should error on text-only endpoint with helpful message
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "text", "text": "What is this?"}
      ]
    }]
  })";
  ExpectError(input, /*is_multimodal=*/false, "multimodal backend");
  ExpectError(input, /*is_multimodal=*/false, "-backend vlm");
}

TEST_F(PreprocessChatJsonTest, MultipleMessagesWithMixedContent) {
  // Multiple messages: some text-only, some with images
  // On multimodal, all arrays are preserved (no combining)
  std::string input = R"({
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": "You are helpful."}]
      },
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
          {"type": "text", "text": "Describe this image"}
        ]
      }
    ]
  })";
  // On multimodal: all arrays preserved unchanged
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

// =============================================================================
// Error handling tests
// =============================================================================

TEST_F(PreprocessChatJsonTest, InvalidJsonReturnsError) {
  std::string input = "not valid json";
  ExpectError(input, /*is_multimodal=*/false, "Invalid JSON");
}

TEST_F(PreprocessChatJsonTest, NonObjectMessageReturnsError) {
  std::string input = R"({"messages": ["not an object"]})";
  ExpectError(input, /*is_multimodal=*/false, "must be an object");
}

TEST_F(PreprocessChatJsonTest, NonObjectContentItemReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": ["not an object"]}]
  })";
  ExpectError(input, /*is_multimodal=*/false, "must be an object");
}

TEST_F(PreprocessChatJsonTest, MissingTextFieldReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": [{"type": "text"}]}]
  })";
  ExpectError(
      input, /*is_multimodal=*/false, "Missing or invalid 'text' field");
}

TEST_F(PreprocessChatJsonTest, NonStringTextFieldReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": [{"type": "text", "text": 123}]}]
  })";
  ExpectError(
      input, /*is_multimodal=*/false, "Missing or invalid 'text' field");
}

TEST_F(PreprocessChatJsonTest, MalformedTextInMultimodalContent) {
  // Multimodal mode skips parsing entirely - validation happens downstream
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "text"}
      ]
    }]
  })";
  // Should pass through unchanged without validation
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_F(PreprocessChatJsonTest, EmptyContentArray) {
  // Empty content array - should result in empty string for non-multimodal
  std::string input = R"({
    "messages": [{"role": "user", "content": []}]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": ""}]
  })";
  ExpectSuccess(input, /*is_multimodal=*/false, expected);
  // For multimodal, empty array is preserved
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, PreservesOtherFields) {
  // Other fields in the request should be preserved
  std::string input = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    "temperature": 0.7,
    "max_tokens": 100
  })";
  std::string expected = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.7,
    "max_tokens": 100
  })";
  ExpectSuccess(input, /*is_multimodal=*/false, expected);
  // For multimodal, array is preserved
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, UnknownContentTypeOnMultimodal) {
  // Unknown content types should pass through on multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "video", "video": {"url": "..."}}]
    }]
  })";
  ExpectSuccess(input, /*is_multimodal=*/true, input);
}

TEST_F(PreprocessChatJsonTest, UnknownContentTypeErrorsOnTextOnly) {
  // Unknown content types should error on text-only with helpful message
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "video", "video": {"url": "..."}}]
    }]
  })";
  ExpectError(input, /*is_multimodal=*/false, "multimodal backend");
}

}  // namespace xllm
