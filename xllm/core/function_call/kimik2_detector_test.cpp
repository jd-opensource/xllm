#include "kimik2_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace llm {
namespace function_call {

class KimiK2DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<KimiK2Detector>();

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

  std::unique_ptr<KimiK2Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

// Test constructor and basic properties
TEST_F(KimiK2DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection
  std::string text_with_tool_call =
      "Some text "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0 "
      "<|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(KimiK2DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<|tool_calls_section_begin|>"));
  EXPECT_TRUE(detector_->has_tool_call(
      "Previous text <|tool_calls_section_begin|>Following content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0 "
      "<|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(
      detector_->has_tool_call("tool_calls_section_begin without brackets"));
  EXPECT_FALSE(
      detector_->has_tool_call("<tool_calls_section_begin without pipes"));
}

// Test trim_whitespace method (indirectly tested through public interface)
TEST_F(KimiK2DetectorTest, TrimWhitespaceHandling) {
  std::string text_with_whitespace =
      "  \t\nPrevious text\r\n  "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": "
      "\"Beijing\"}<|tool_call_end|><|tool_calls_section_end|>  \t\r\n";

  auto result = detector_->detect_and_parse(text_with_whitespace, tools_);

  // Verify normal text is correctly extracted
  EXPECT_EQ(result.normal_text, "  \t\nPrevious text\r\n  ");

  // Verify tool call is correctly parsed
  EXPECT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].tool_index, 0);
}

// Test single tool call parsing
TEST_F(KimiK2DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": \"Beijing\", "
      "\"unit\": \"celsius\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["unit"], "celsius");
}

// Test multiple tool calls parsing
TEST_F(KimiK2DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate an expression "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": "
      "\"Shanghai\"}<|tool_call_end|><|tool_call_begin|>functions.calculate:1 "
      "<|tool_call_argument_begin|>{\"expression\": \"2 + 3 * "
      "4\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate an expression ");
  EXPECT_EQ(result.calls.size(), 2);

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, 0);
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_current_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["location"], "Shanghai");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, 1);
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "calculate");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["expression"], "2 + 3 * 4");
}

// Test invalid JSON handling
TEST_F(KimiK2DetectorTest, InvalidJsonHandling) {
  std::string text =
      "Test invalid JSON "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": \"Beijing\", "
      "invalid_json}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test invalid JSON ");
  // KimiK2 detector should still parse the call even with invalid JSON, leaving
  // JSON validation to higher levels
  EXPECT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
}

// Test empty tool call content
TEST_F(KimiK2DetectorTest, EmptyToolCallContent) {
  std::string text =
      "Test empty content "
      "<|tool_calls_section_begin|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content ");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call (only start tag)
TEST_F(KimiK2DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call ");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(KimiK2DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.unknown_tool:0 "
      "<|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool ");
  // KimiK2 detector should parse the call regardless of whether the tool is
  // known
  EXPECT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value(), "unknown_tool");
}

// Test case with only normal text
TEST_F(KimiK2DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(KimiK2DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test whitespace-only input
TEST_F(KimiK2DetectorTest, WhitespaceOnlyInput) {
  std::string text = "   \t\n\r   ";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "   \t\n\r   ");
  EXPECT_EQ(result.calls.size(), 0);
}

// Test complex nested JSON parameters
TEST_F(KimiK2DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": \"Beijing\", "
      "\"options\": {\"include_forecast\": true, \"days\": 7, \"details\": "
      "[\"temperature\", \"humidity\", "
      "\"wind\"]}}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
  EXPECT_EQ(params["options"]["details"].size(), 3);
}

// Test tool call in the middle of text
TEST_F(KimiK2DetectorTest, ToolCallInMiddleOfText) {
  std::string text =
      "Previous text "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.calculate:0 "
      "<|tool_call_argument_begin|>{\"expression\": "
      "\"1+1\"}<|tool_call_end|><|tool_calls_section_end|> Following text";

  auto result = detector_->detect_and_parse(text, tools_);

  // Note: According to KimiK2 implementation, only text before tool call
  // section is preserved as normal_text
  EXPECT_EQ(result.normal_text, "Previous text ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);
  EXPECT_EQ(call.name.value(), "calculate");
}

// Test special characters handling
TEST_F(KimiK2DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": \"New York City\", "
      "\"note\": \"Contains "
      "symbols！@#$%^&*()_+=\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test function name extraction
TEST_F(KimiK2DetectorTest, FunctionNameExtraction) {
  std::string text =
      "Function name test "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.my_custom_"
      "function:5 <|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Function name test ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 5);  // Should extract the index correctly
  EXPECT_EQ(call.name.value(), "my_custom_function");
}

// Test malformed function ID handling
TEST_F(KimiK2DetectorTest, MalformedFunctionIdHandling) {
  std::string text =
      "Malformed ID test "
      "<|tool_calls_section_begin|><|tool_call_begin|>invalid_format "
      "<|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Malformed ID test ");
  // Malformed format that doesn't match regex should result in no calls
  EXPECT_EQ(result.calls.size(), 0);
}

// Test malformed function ID that matches regex but has invalid format
TEST_F(KimiK2DetectorTest, MalformedButMatchingFunctionId) {
  std::string text =
      "Malformed but matching test "
      "<|tool_calls_section_begin|><|tool_call_begin|>invalid.format:0 "
      "<|tool_call_argument_begin|>{\"param\": "
      "\"value\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Malformed but matching test ");
  // Should parse but with empty function name due to missing "functions."
  // prefix
  EXPECT_EQ(result.calls.size(), 1);
  const auto& call = result.calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "");  // Empty function name for malformed ID
  EXPECT_EQ(call.tool_index, 0);
}

// Test multiple sections (edge case)
TEST_F(KimiK2DetectorTest, MultipleSections) {
  std::string text =
      "First section "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": "
      "\"Beijing\"}<|tool_call_end|><|tool_calls_section_end|> Middle text "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.calculate:1 "
      "<|tool_call_argument_begin|>{\"expression\": "
      "\"1+1\"}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  // Should extract text before first section
  EXPECT_EQ(result.normal_text, "First section ");
  // Should parse all tool calls from all sections
  EXPECT_EQ(result.calls.size(), 2);

  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
  EXPECT_EQ(result.calls[1].name.value(), "calculate");
}

// Performance test: many tool calls
TEST_F(KimiK2DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test <|tool_calls_section_begin|>";

  // Build text containing multiple tool calls
  for (int i = 0; i < 10000; ++i) {
    text += "<|tool_call_begin|>functions.calculate:" + std::to_string(i) +
            " <|tool_call_argument_begin|>{\"expression\": \"" +
            std::to_string(i) + " + " + std::to_string(i + 1) +
            "\"}<|tool_call_end|>";
  }
  text += "<|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test ");
  EXPECT_EQ(result.calls.size(), 10000);

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 10000; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, i);
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

// Test edge case: nested braces in JSON
TEST_F(KimiK2DetectorTest, NestedBracesInJson) {
  std::string text =
      "Nested braces test "
      "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_"
      "weather:0 <|tool_call_argument_begin|>{\"location\": \"Beijing\", "
      "\"config\": {\"nested\": {\"deep\": "
      "\"value\"}}}<|tool_call_end|><|tool_calls_section_end|>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Nested braces test ");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);
  EXPECT_EQ(call.name.value(), "get_current_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

}  // namespace function_call
}  // namespace llm