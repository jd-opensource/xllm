#include "qwen25_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace llm {
namespace function_call {

class Qwen25DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Qwen25Detector>();

    // Setup test tools
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"location",
           {{"type", "string"},
            {"description", "The city and state, e.g. San Francisco, CA"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
        {"required", {"location"}}};

    JsonFunction weather_func("get_current_weather",
                              "Get the current weather in a given location",
                              weather_params);
    weather_tool_ = JsonTool("function", weather_func);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);
    calculator_tool_ = JsonTool("function", calculator_func);

    tools_ = {weather_tool_, calculator_tool_};
  }

  std::unique_ptr<Qwen25Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

// Test constructor and basic properties
TEST_F(Qwen25DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection
  std::string text_with_tool_call =
      "Some text <tool_call>\n{\"name\": \"test\"}\n</tool_call>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(Qwen25DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>\n"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>\nFollowing content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>\n{\"name\": \"test\"}\n</tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without newline"));
}

// Test trim_whitespace method (indirectly tested through public interface)
TEST_F(Qwen25DetectorTest, TrimWhitespaceHandling) {
  std::string text_with_whitespace =
      "  \t\nPrevious text\r\n  <tool_call>\n  {\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"Beijing\"}}  "
      "\n</tool_call>  \t\r\n";

  auto result = detector_->detect_and_parse(text_with_whitespace, tools_);

  // Verify normal text is correctly trimmed
  EXPECT_EQ(result.normal_text, "Previous text");

  // Verify tool call is correctly parsed
  EXPECT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].tool_index, -1);  // Base class always returns -1
}

// Test single tool call parsing
TEST_F(Qwen25DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather <tool_call>\n{\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"Beijing\", "
      "\"unit\": \"celsius\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["unit"], "celsius");
}

// Test multiple tool calls parsing
TEST_F(Qwen25DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate an expression "
      "<tool_call>\n{\"name\": \"get_current_weather\", \"arguments\": "
      "{\"location\": \"Shanghai\"}}\n</tool_call>\n<tool_call>\n{\"name\": "
      "\"calculate\", \"arguments\": {\"expression\": \"2 + 3 * "
      "4\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate an expression");
  EXPECT_EQ(result.calls.size(), 2);

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_current_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["location"], "Shanghai");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "calculate");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["expression"], "2 + 3 * 4");
}

// Test invalid JSON handling
TEST_F(Qwen25DetectorTest, InvalidJsonHandling) {
  std::string text =
      "Test invalid JSON <tool_call>\n{\"name\": \"get_current_weather\", "
      "\"arguments\": {\"location\": \"Beijing\", invalid_json}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test invalid JSON");
  EXPECT_EQ(result.calls.size(), 0);  // Invalid JSON should be ignored
}

// Test empty tool call content
TEST_F(Qwen25DetectorTest, EmptyToolCallContent) {
  std::string text = "Test empty content <tool_call>\n   \t\n  \n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call (only start tag)
TEST_F(Qwen25DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call <tool_call>\n{\"name\": \"get_current_weather\"";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Qwen25DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool <tool_call>\n{\"name\": \"unknown_tool\", \"arguments\": "
      "{\"param\": \"value\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools, so should be 0 calls
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(Qwen25DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(Qwen25DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test whitespace-only input
TEST_F(Qwen25DetectorTest, WhitespaceOnlyInput) {
  std::string text = "   \t\n\r   ";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
}

// Test complex nested JSON parameters
TEST_F(Qwen25DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test <tool_call>\n{\"name\": \"get_current_weather\", "
      "\"arguments\": {\"location\": \"Beijing\", \"options\": "
      "{\"include_forecast\": true, \"days\": 7, \"details\": "
      "[\"temperature\", \"humidity\", \"wind\"]}}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
  EXPECT_EQ(params["options"]["details"].size(), 3);
}

// Test tool call in the middle of text
TEST_F(Qwen25DetectorTest, ToolCallInMiddleOfText) {
  std::string text =
      "Previous text <tool_call>\n{\"name\": \"calculate\", \"arguments\": "
      "{\"expression\": \"1+1\"}}\n</tool_call> Following text";

  auto result = detector_->detect_and_parse(text, tools_);

  // Note: According to implementation, only text before tool call is preserved
  // as normal_text
  EXPECT_EQ(result.normal_text, "Previous text");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_EQ(call.name.value(), "calculate");
}

// Test special characters handling
TEST_F(Qwen25DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test <tool_call>\n{\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"New York "
      "City\", \"note\": \"Contains symbols！@#$%^&*()_+=\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Performance test: many tool calls
TEST_F(Qwen25DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test";

  // Build text containing multiple tool calls
  for (int i = 0; i < 10000; ++i) {
    text +=
        " <tool_call>\n{\"name\": \"calculate\", \"arguments\": "
        "{\"expression\": \"" +
        std::to_string(i) + " + " + std::to_string(i + 1) +
        "\"}}\n</tool_call>";
  }

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test");
  EXPECT_EQ(result.calls.size(), 10000);

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 10000; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

}  // namespace function_call
}  // namespace llm