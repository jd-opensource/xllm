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

#include "glm47_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace function_call {

class Glm47DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Glm47Detector>();

    // Setup test tools
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city name, e.g. Beijing, Shanghai"}}},
          {"date",
           {{"type", "string"},
            {"description", "Date in YYYY-MM-DD format"}}}}},
        {"required", {"city"}}};

    JsonFunction weather_func("get_weather",
                              "Get the weather information for a given city",
                              weather_params);
    weather_tool_ = JsonTool("function", weather_func);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}},
          {"precision",
           {{"type", "number"}, {"description", "Number of decimal places"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);
    calculator_tool_ = JsonTool("function", calculator_func);

    tools_ = {weather_tool_, calculator_tool_};
  }

  std::unique_ptr<Glm47Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

// Test constructor and basic properties
TEST_F(Glm47DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection (GLM-4.7 compact format)
  std::string text_with_tool_call =
      "Some text "
      "<tool_call>test<arg_key>param</arg_key><arg_value>value</arg_value></"
      "tool_call>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(Glm47DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>Following content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</"
      "arg_value></tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without closing"));
}

// Test single tool call parsing (GLM-4.7 compact format)
TEST_F(Glm47DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test multiple tool calls parsing (GLM-4.7 format)
TEST_F(Glm47DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>"
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate");
  ASSERT_EQ(result.calls.size(), 2);  // Use ASSERT to stop test if this fails

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, -1);
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["city"], "上海");
  EXPECT_EQ(params1["date"], "2024-06-27");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, -1);
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "get_weather");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["city"], "北京");
  EXPECT_EQ(params2["date"], "2024-06-27");
}

// Test GLM-4.7 specific compact format
TEST_F(Glm47DetectorTest, Glm47CompactFormat) {
  std::string text =
      "Weather query "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Weather query");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
}

// Test number type coercion
TEST_F(Glm47DetectorTest, NumberTypeCoercion) {
  std::string text =
      "Calculate with precision "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>3.14 * "
      "2</arg_value>"
      "<arg_key>precision</arg_key><arg_value>2</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Calculate with precision");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "calculate");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["expression"], "3.14 * 2");
  // precision should be parsed as number
  EXPECT_TRUE(params["precision"].is_number());
  EXPECT_EQ(params["precision"], 2);
}

// Test empty tool call content
TEST_F(Glm47DetectorTest, EmptyToolCallContent) {
  std::string text = "Test empty content <tool_call>test</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call
TEST_F(Glm47DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Glm47DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool "
      "<tool_call>unknown_tool<arg_key>param</arg_key><arg_value>value</"
      "arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(Glm47DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(Glm47DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test complex nested JSON parameters
TEST_F(Glm47DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>options</arg_key><arg_value>{\"include_forecast\": true, "
      "\"days\": 7}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
}

// Test special characters handling
TEST_F(Glm47DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York "
      "City</arg_value>"
      "<arg_key>note</arg_key><arg_value>Contains "
      "symbols！@#$%^&*()_+=</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test whitespace handling in arg values
TEST_F(Glm47DetectorTest, WhitespaceHandlingInArgValues) {
  std::string text =
      "Whitespace test "
      "<tool_call>get_weather<arg_key>  city  </arg_key><arg_value>  Beijing  "
      "</arg_value>"
      "<arg_key>\t\ndate\r\n</arg_key><arg_value>\n\t2024-06-27\r\n</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Whitespace test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");  // Whitespace should be trimmed
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test streaming parsing functionality
TEST_F(Glm47DetectorTest, StreamingParseBasicFunctionality) {
  std::string chunk1 = "<tool_call>get_weather";
  std::string chunk2 = "<arg_key>city</arg_key><arg_value>";
  std::string chunk3 = "Beijing</arg_value></tool_call>";

  // First chunk - should send tool name
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.calls.size(), 1);
  EXPECT_TRUE(result1.calls[0].name.has_value());
  EXPECT_EQ(result1.calls[0].name.value(), "get_weather");

  // Second chunk - should stream partial JSON
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_GE(result2.calls.size(), 0);  // May have partial JSON output

  // Third chunk - completes the tool call
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 0);  // Should complete the JSON
}

// Test streaming parse with normal text
TEST_F(Glm47DetectorTest, StreamingParseWithNormalText) {
  std::string chunk1 = "Please check the weather ";
  std::string chunk2 = "<tool_call>get_weather";
  std::string chunk3 =
      "<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>";

  // First chunk - normal text should be returned
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.normal_text, "Please check the weather ");
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - tool call start, should send tool name
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_EQ(result2.calls.size(), 1);
  EXPECT_EQ(result2.calls[0].name.value(), "get_weather");

  // Third chunk - complete tool call
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 0);
}

// Test invalid JSON in arg values
TEST_F(Glm47DetectorTest, InvalidJsonInArgValues) {
  std::string text =
      "Invalid JSON test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{invalid json}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Invalid JSON test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"], "{invalid json}");  // Should be treated as string
}

// Test nested braces in JSON values
TEST_F(Glm47DetectorTest, NestedBracesInJsonValues) {
  std::string text =
      "Nested braces test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{\"nested\": {\"deep\": "
      "\"value\"}}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Nested braces test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

// Test performance with many tool calls
TEST_F(Glm47DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test ";

  // Build text containing multiple tool calls
  for (int i = 0; i < 100; ++i) {
    text += "<tool_call>calculate<arg_key>expression</arg_key><arg_value>" +
            std::to_string(i) + " + " + std::to_string(i + 1) +
            "</arg_value></tool_call>";
  }

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test");
  ASSERT_EQ(result.calls.size(), 100);  // Use ASSERT to stop test if this fails

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 100; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, -1);
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

}  // namespace function_call
}  // namespace xllm
